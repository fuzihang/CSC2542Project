f = open('control_rnn_out.txt', 'r')
a = []
b = []

for line in f.readlines():
    if 'Current evaluation' in line:
        a.append(-1 * float(line.split(':')[1]))


b = [3 * i for i in range(len(a))]

import matplotlib.pyplot as plt

plt.plot(b, a)
plt.xlabel('CMA-ES generations')
plt.ylabel('Average reward collected for 100 rollouts')
plt.show()
