import sys
import os
import numpy as np

if len(sys.argv) != 3:
    print('usage: %s plda_txt output_dir' % sys.argv[0])
    quit()

plda_txt = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if os.path.isfile(os.path.join(output_dir, "mean.txt")):
    os.remove(os.path.join(output_dir, "mean.txt"))

if os.path.isfile(os.path.join(output_dir, "psi.txt")):
    os.remove(os.path.join(output_dir, "psi.txt"))

if os.path.isfile(os.path.join(output_dir, "transform.txt")):
    os.remove(os.path.join(output_dir, "transform.txt"))

with open(plda_txt, 'r') as f:
    line = f.readline()
    mean = line.strip().split('[')[1].split(']')[0].strip().split(' ')
    mean = [ float(num) for num in mean ]
    np.savetxt(os.path.join(output_dir, "mean.txt"), mean)

    f.readline()
    lines = f.readlines()
    psi = lines[-2].strip().split('[')[1].split(']')[0].strip().split(' ')
    psi = [ float(num) for num in psi]
    np.savetxt(os.path.join(output_dir, "psi.txt"), psi)

    transform_lines = lines[:-2]
    transform_lines[-1] = transform_lines[-1].split(']')[0]
    transform = []
    for l in transform_lines:
        transform.append([float(num) for num in l.strip().split(' ')])
    transform = np.array(transform)
    np.savetxt(os.path.join(output_dir, "transform.txt"), transform)

    assert(len(psi) == transform.shape[0] and transform.shape[1] == len(mean))
