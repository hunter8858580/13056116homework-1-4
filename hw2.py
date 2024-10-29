from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2

plt.rcParams['figure.dpi'] = 144 

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

# 讀取影像檔, 並保留亮度成分
img = cv2.imread('data/svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)

# convert img to float data type
A = img.astype(dtype=np.float64)

# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T


def compute_energy(X: np.ndarray):
    # return energy of X
    # For more details on the energy of a 2D signal, see the 
    # class notebook: 內容庫/補充說明/Energy of a 2D Signal.
    return np.sum(X ** 2)

# img_h and img_w are image's height and width, respectively
img_h, img_w = A.shape
# Compute SNR
keep_r = 201
rs = np.arange(1, keep_r)

# compute energy of A, and save it to variable Energy_A
energy_A = compute_energy(A)

# Decalre an array to save the energy of noise vs r.
# energy_N[r] is the energy of A - A_bar(sum of the first r components)
energy_N = np.zeros(keep_r) # energy_N[0]棄置不用

for r in rs:
    # A_bar is the sum of the first r components of SVD
    # A_bar is an approximation of A
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :] 
    Noise = A - A_bar 
    energy_N[r] = compute_energy(Noise) 

# 計算 SNR 和作圖
SNR = 10 * np.log10(energy_A / energy_N[1:])

# 繪製 SNR 與 r 的關係圖
plt.figure()
plt.plot(rs, SNR)
plt.xlabel('r (Number of Components)')
plt.ylabel('SNR (dB)')
plt.title('SNR vs Number of Components')
plt.grid(True)
plt.show()

# --------------------------
# Verify that energy_N[r] equals the sum of lambda_i, i from r+1 to n,
# lambda_i is the eigenvalue of A^T @ A

eigenvalues, _ = myeig(A.T @ A, symmetric=True)
eigenvalues = np.real(eigenvalues)

# 計算每個 r 對應的理論能量
energy_sum = np.zeros(keep_r)
for r in rs:
    energy_sum[r] = np.sum(eigenvalues[r:])

# 繪製 energy_N 和理論能量的比較圖
plt.figure()
plt.plot(rs, energy_N[1:], label='Computed Noise Energy')
plt.plot(rs, energy_sum[1:], label='Theoretical Energy (Sum of Eigenvalues)')
plt.xlabel('r (Number of Components)')
plt.ylabel('Energy')
plt.title('Energy of Noise vs Theoretical Sum of Eigenvalues')
plt.legend()
plt.grid(True)
plt.show()
