import sys
import os
import numpy as np

if len(sys.argv) != 5:
    print('usage: %s mu transform psi plda' % sys.argv[0])
    quit()

mu = sys.argv[1]
transform = sys.argv[2]
psi = sys.argv[3]
plda = sys.argv[4]

mu = np.loadtxt(mu)
transform = np.loadtxt(transform)
psi = np.loadtxt(psi)

with open(plda, 'w') as f:
    f.write('<Plda>  [ ')
    for i in range(mu.shape[0]):
        f.write('%f ' % mu[i])
    f.write(']\n [\n')
    for i in range(transform.shape[0]):
        f.write('  ')
        for j in range(transform.shape[1]):
            f.write('%f ' % transform[i, j])
        if i == (transform.shape[0] - 1):
            f.write(']')
        f.write('\n')
    f.write(' [ ')
    for i in range(psi.shape[0]):
        f.write('%f ' % psi[i])
    f.write(']\n</Plda> ')
