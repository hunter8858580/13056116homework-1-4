import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cost function
def cost_function(w, x, y):
    predictions = w[0] + w[1] * np.sin(w[2] * x + w[3])
    errors = y - predictions
    return np.sum(errors ** 2)

# Scatter plot range function
def scatter_pts_2d(x, y):
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin, xmax, ymin, ymax

# Load dataset
file_path = r'C:\Users\hunte\Desktop\資訊科技\data\hw7.csv'
dataset = pd.read_csv(file_path).to_numpy(dtype=np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# Initialize parameters
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
alpha = 0.05
max_iters = 500
epsilon = 1e-8  # For numerical gradient calculation

# Analytic gradient descent
for _ in range(max_iters):
    predictions = w[0] + w[1] * np.sin(w[2] * x + w[3])
    errors = y - predictions
    
    grad_w0 = -2 * np.sum(errors)
    grad_w1 = -2 * np.sum(errors * np.sin(w[2] * x + w[3]))
    grad_w2 = -2 * np.sum(errors * w[1] * x * np.cos(w[2] * x + w[3]))
    grad_w3 = -2 * np.sum(errors * w[1] * np.cos(w[2] * x + w[3]))
    
    gradient = np.array([grad_w0, grad_w1, grad_w2, grad_w3])
    w -= alpha * gradient

# Generate predictions for analytic method
xmin, xmax = np.min(x), np.max(x)
xgap = (xmax - xmin) * 0.2
xmin -= xgap
xmax += xgap
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# Reset parameters for numeric gradient descent
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

# Numerical gradient descent
for _ in range(max_iters):
    grad_numeric = np.zeros_like(w)
    for i in range(len(w)):
        w_temp = w.copy()
        w_temp[i] += epsilon
        grad_numeric[i] = (cost_function(w_temp, x, y) - cost_function(w, x, y)) / epsilon
    
    w -= alpha * grad_numeric

# Generate predictions for numeric method
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# Plot x vs y, xt vs yt1, xt vs yt2
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin, xmax])
plt.ylim([np.min(y) - 0.5, np.max(y) + 0.5])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
