import numpy as np 
import matplotlib.pyplot as plt 


x1 = np.linspace(1,10,10)
y1 = np.linspace(1,10,10)
x2 = np.random.normal(2, 1.2, 300)
y2 = np.random.normal(2, 1.2, 300)
x3 = np.random.normal(7, 1.2, 300)
y3 = np.random.normal(7, 1.2, 300)


fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(x1, y1, c='b', label='Prediction')
ax1.scatter(x1, y1, c='r', label='Training Set')
ax1.legend()

ax2.scatter(x2, y2, c='g', label='Class 1')
ax2.scatter(x3, y3, c='y', label='Class 2')
ax2.legend()

plt.show()