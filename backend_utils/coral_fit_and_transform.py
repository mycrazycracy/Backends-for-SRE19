import sys
import os
import numpy as np
import kaldi_io

if len(sys.argv) != 4:
    print("usage: %s ood-xvectors id-xvectors ood-transformed-dir" % sys.argv[0])
    print("The x-vectors should be normalized by the mean.")
    quit()

print("Perform CORAL transform")
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
Cs = (1.0 / ood_vec.shape[0]) * np.dot(np.transpose(ood_vec), ood_vec) + np.eye(dim)
Ct = (1.0 / id_vec.shape[0]) * np.dot(np.transpose(id_vec), id_vec) + np.eye(dim)

# Compute Cs^(-1/2)
ws, vs = np.linalg.eig(Cs)
whitening = np.dot(np.dot(vs, np.diag(ws**(-0.5))), np.transpose(vs))

wt, vt = np.linalg.eig(Ct)
coloring = np.dot(np.dot(vt, np.diag(wt**(0.5))), np.transpose(vt))

# Transform the out-of-domain vectors
transform_ood_vec = np.dot(np.dot(ood_vec, whitening), coloring)

if not os.path.isdir(transform_dir):
    os.makedirs(transform_dir)

with open(os.path.join(transform_dir, 'xvector_transformed.ark'), 'wb') as f:
    for index, key in enumerate(ood_keys):
        kaldi_io.write_vec_flt(f, transform_ood_vec[index], key=key)
        
