import subprocess
import multiprocessing

processes = 10

def work(i):
    cmd = f'python3 run_doom_take_cover.py {1000 * i} {1000 * (i+1)}'
    print(cmd)
    subprocess.call(cmd)

with multiprocessing.Pool(processes) as p:
    p.map(work, range(processes))
