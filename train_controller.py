"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from controller import Controller
import numpy as np
from misc import RolloutGenerator, ACTION_DIM, RNN_HIDDEN_DIM, VAE_LATENT_DIM
from misc import load_parameters
from utils import str2bool
from misc import flatten_parameters
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--use_rnn', type=str, required=True)
args = parser.parse_args()

use_rnn = str2bool(args.use_rnn)

assert type(use_rnn) is bool

# Multiprocessing
num_samples = 64
num_solutions = 16
num_workers = 6
time_limit = 1000
dr = 'temp'
target_return = 700

# create tmp dir if non existent and clean it if existent
tmp_dir = join(dr, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# create ctrl dir if non exitent
ctrl_dir = join(dr, 'ctrl')
ctrl_dir_rnn = join(dr, 'ctrl_rnn')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

if not exists(ctrl_dir_rnn):
    mkdir(ctrl_dir_rnn)

vae_file = 'vae_final.weights'
rnn_file = 'log/mdrnn/checkpoint.tar'
ctrl_file = 'temp/ctrl_rnn/best.tar' if use_rnn else 'temp/ctrl/best.tar'


################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    if torch.cuda.is_available():
        gpu = p_index % torch.cuda.device_count()
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(vae_file, ctrl_file, rnn_file, device, time_limit, use_rnn=use_rnn)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


################################################################################
#                Define queues and start workers                               #
################################################################################
p_queue = Queue()
r_queue = Queue()
e_queue = Queue()

for p_index in range(num_workers):
    Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()
    # slave_routine(p_queue, r_queue, e_queue, p_index)


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(solutions, results, rollouts=100):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)

################################################################################
#                           Launch CMA                                         #
################################################################################
if use_rnn:
    controller = Controller(VAE_LATENT_DIM + RNN_HIDDEN_DIM, ACTION_DIM)
else:
    controller = Controller(VAE_LATENT_DIM, ACTION_DIM)


# define current best and load parameters
cur_best = None
print("Attempting to load previous best...")
if exists(ctrl_file):
    state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
    cur_best = - state['reward']
    controller.load_state_dict(state['state_dict'])
    print("Previous best was {}...".format(-cur_best))

parameters = controller.parameters()
es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                              {'popsize': num_solutions})

epoch = 0
log_step = 3
while not es.stop():
    if cur_best is not None and cur_best > target_return:
        print("Already better than target, breaking...")
        break

    r_list = [0] * num_solutions  # result list
    solutions = es.ask()

    # push parameters to queue
    for s_id, s in enumerate(solutions):
        for _ in range(num_samples):
            p_queue.put((s_id, s))

    # retrieve results
    pbar = tqdm(total=num_solutions * num_samples)
    for _ in range(num_solutions * num_samples):
        while r_queue.empty():
            sleep(.1)
        r_s_id, r = r_queue.get()
        r_list[r_s_id] += r / num_samples
        pbar.update(1)
    pbar.close()

    es.tell(solutions, r_list)
    es.logger.add()
    es.disp()

    # evaluation and saving
    if epoch % log_step == log_step - 1:
        best_params, best, std_best = evaluate(solutions, r_list)
        print("Current evaluation: {}".format(best))
        if not cur_best or cur_best > best:
            cur_best = best
            print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
            load_parameters(best_params, controller)
            torch.save(
                {'epoch': epoch,
                 'reward': - cur_best,
                 'state_dict': controller.state_dict()},
                join(ctrl_dir_rnn if use_rnn else ctrl_dir, 'best.tar'))
        if -best > target_return:
            print("Terminating controller training with value {}...".format(best))
            break


    epoch += 1

es.result_pretty()
e_queue.put('EOP')
cma.plot()

