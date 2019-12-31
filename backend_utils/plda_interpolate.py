import sys
import os
import numpy as np

if len(sys.argv) != 5:
    print('usage: %s alpha id_plda_dir ood_plda_dir output_plda_dir' % sys.argv[0])
    quit()

alpha = float(sys.argv[1])
id_dir = sys.argv[2]
ood_dir = sys.argv[3]
output_dir = sys.argv[4]

id_mean = np.loadtxt(os.path.join(id_dir, 'mean.txt'))
id_trans = np.loadtxt(os.path.join(id_dir, 'transform.txt'))
id_psi = np.loadtxt(os.path.join(id_dir, 'psi.txt'))

ood_mean = np.loadtxt(os.path.join(ood_dir, 'mean.txt'))
ood_trans = np.loadtxt(os.path.join(ood_dir, 'transform.txt'))
ood_psi = np.loadtxt(os.path.join(ood_dir, 'psi.txt'))

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Get the ID covariance
id_A_inv = np.linalg.inv(id_trans)
id_AT_inv = np.linalg.inv(np.transpose(id_trans))
id_Psi = np.diag(id_psi)
id_W = np.dot(id_A_inv, id_AT_inv)
id_B = np.dot(np.dot(id_A_inv, id_Psi), id_AT_inv)

# Get the OOD covariance
ood_A_inv = np.linalg.inv(ood_trans)
ood_AT_inv = np.linalg.inv(np.transpose(ood_trans))
ood_Psi = np.diag(ood_psi)
ood_W = np.dot(ood_A_inv, ood_AT_inv)
ood_B = np.dot(np.dot(ood_A_inv, ood_Psi), ood_AT_inv)

new_W = alpha * id_W + (1-alpha) * ood_W
new_B = alpha * id_B + (1-alpha) * ood_B

# Use the ID mean
new_mean = id_mean

# # Use the interpolated mean
# new_mean = alpha * id_mean + (1-alpha) * ood_mean

# Convert to A and Psi
w, v = np.linalg.eig(new_W)
W = np.dot(v, np.diag(w ** (-0.5)))
e, p = np.linalg.eig(np.dot(np.dot(np.transpose(W), new_B), W))
B = np.dot(W, p)

# return B, e
ind = np.argsort(e)
new_trans = np.transpose(B[:, ind[::-1]])
new_psi = e[ind[::-1]]

# Write the PLDA
np.savetxt(os.path.join(output_dir, "mean.txt"), new_mean)
np.savetxt(os.path.join(output_dir, "transform.txt"), new_trans)
np.savetxt(os.path.join(output_dir, "psi.txt"), new_psi)

