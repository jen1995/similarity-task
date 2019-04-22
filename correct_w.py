import numpy as np
from tqdm import tqdm

W = []
with open('model_neg_new.vec') as f:
    l = next(f)
    for line in tqdm(f, desc='reading model'):
        vector = list(map(float, line.split()[1:]))
        W.append(vector)

W = np.array(W)
with open('output_new.txt') as f:
     D = f.readlines()
D = np.array(D[0].split(), dtype=np.float64)
D = np.diag(D)
WtW = W.T @ W
L = np.linalg.cholesky(WtW)

A = np.linalg.inv(L).T
assert np.allclose(A.T @ WtW @ A, np.eye(WtW.shape[0], WtW.shape[0]))

M = A.T @ D @ A

values, vectors = np.linalg.eigh(M)
new_values = np.zeros(D.shape[0])
new_values[np.diagonal(D) < 0] = values[values < 0]
new_values[np.diagonal(D) > 0] = values[values > 0]

new_vectors = np.zeros_like(vectors)
new_vectors[:, np.diagonal(D) < 0] = vectors[:, values < 0]
new_vectors[:, np.diagonal(D) > 0] = vectors[:, values > 0]


Lambda = np.diag(1 / np.abs(new_values))
S = A @ new_vectors @ np.sqrt(Lambda)

assert np.allclose(S @ D @ S.T, D)
assert np.allclose(S.T @ WtW @ S, Lambda, atol=1e-05)

new_W = W @ S

with open('model_neg_new.vec') as f:
    with open('model_neg_corrected_new.vec', 'w') as f_cor:
        first_line = next(f)
        print(first_line, file=f_cor, end="")
        counter = 0
        for line in tqdm(f, desc='writing the corrected W to model_neg_corrected.vec'):
            word = line.split()[0]
            print(word, file=f_cor, end=" ")
            print(*new_W[counter], sep=" ", file=f_cor)
            counter += 1
