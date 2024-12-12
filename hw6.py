import numpy as np
import matplotlib.pyplot as plt

# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2], [0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2], [0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# Compute means
m1 = np.mean(X1, axis=0, keepdims=True)
m2 = np.mean(X2, axis=0, keepdims=True)

# Compute within-class scatter matrix
S1 = np.cov(X1, rowvar=False)
S2 = np.cov(X2, rowvar=False)
Sw = S1 + S2

# Compute linear discriminant direction
w = np.linalg.inv(Sw) @ (m2 - m1).T
w = w.flatten()  # Convert to 1D array

# Normalize w
w = w / np.linalg.norm(w)

# Project points onto w
projection_X1 = (X1 @ w)[:, None] * w  # Projection of X1 on w
projection_X2 = (X2 @ w)[:, None] * w  # Projection of X2 on w

# Plot original points
plt.figure(dpi=144)
plt.plot(X1[:, 0], X1[:, 1], 'r.', label='Class 1')
plt.plot(X2[:, 0], X2[:, 1], 'g.', label='Class 2')

# Plot projections
plt.plot(projection_X1[:, 0], projection_X1[:, 1], 'ro', alpha=0.5, label='Class 1 Projection')
plt.plot(projection_X2[:, 0], projection_X2[:, 1], 'go', alpha=0.5, label='Class 2 Projection')

# Draw the discriminant line
line_points = np.linspace(-2, 5, 100)
slope = w[1] / w[0]
line = slope * line_points
plt.plot(line_points, line, 'b-', label='LDA Line')

plt.axis('equal')
plt.legend()
plt.show()

