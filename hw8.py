from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = r'C:\Users\hunte\Desktop\資訊科技\data\hw8.csv'
hw8_csv = pd.read_csv(file_path)
hw8_dataset = hw8_csv.to_numpy(dtype=np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

# Fit SVM model
svm_model = SVC(kernel='rbf', C=1, gamma=0.5)
svm_model.fit(X0, y)

# Generate grid for decision boundary visualization
x_min, x_max = X0[:, 0].min() - 1, X0[:, 0].max() + 1
y_min, y_max = X0[:, 1].min() - 1, X0[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict classifications for the grid points
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and data points
fig = plt.figure(dpi=288)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.title("SVM with RBF Kernel")
plt.show()
