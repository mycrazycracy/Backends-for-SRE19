import sys
import os
import numpy as np
import kaldi_io

if len(sys.argv) != 4:
    print("usage: %s ood-xvectors id-xvectors ood-transformed-dir" % sys.argv[0])
    print("The x-vectors should be normalized by the mean.")
    quit()

print("Perform FDA transform")
ood_vec_file = sys.argv[1]
id_vec_file = sys.argv[2]
transform_dir = sys.argv[3]

ood_vec = []
ood_keys = []
for key, vec in kaldi_io.read_vec_flt_scp(ood_vec_file):
    ood_vec.append(vec)
    ood_keys.append(key)
ood_vec = np.array(ood_vec)

id_vec = []
for key, vec in kaldi_io.read_vec_flt_scp(id_vec_file):
    id_vec.append(vec)
id_vec = np.array(id_vec)
dim = id_vec.shape[1]

# Covariance
# No need to add the identity as CORAL
Cs = (1.0 / ood_vec.shape[0]) * np.dot(np.transpose(ood_vec), ood_vec)
Ct = (1.0 / id_vec.shape[0]) * np.dot(np.transpose(id_vec), id_vec)

# Compute Cs^(-1/2)
ws, vs = np.linalg.eig(Cs)
Csp = np.dot(np.dot(vs, np.diag(ws**(0.5))), np.transpose(vs))
Csn = np.dot(np.dot(vs, np.diag(ws**(-0.5))), np.transpose(vs))
C = np.dot(np.dot(Csn, Ct), Csn)

w, v = np.linalg.eig(C)
w_floor = np.maximum(w, 1)

# Transform the out-of-domain vectors
transform_ood_vec = np.dot(ood_vec, np.transpose(np.dot(np.dot(Csp, np.dot(np.dot(v, np.diag(w_floor**(0.5))), np.transpose(v))), Csn)))

if not os.path.isdir(transform_dir):
    os.makedirs(transform_dir)

with open(os.path.join(transform_dir, 'xvector_transformed.ark'), 'w') as f:
    for index, key in enumerate(ood_keys):
        kaldi_io.write_vec_flt(f, transform_ood_vec[index], key=key)
        

