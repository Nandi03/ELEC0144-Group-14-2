import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-1, 1, 0.05)
l = len(x)
d = 0.8* x**3 + 0.3 * x**2 - 0.4*x + np.random.normal(0, 0.02, l)

plt.plot(x, d)


# Settings

w0 = 0.5
w1 = 0.4
w2 = 0.2

eta = 0.01
iteration = 500

# Initial fit

ytestx0 = w0*(-1)**3 + w1*(-1)**2 - w2*(-1)
ytestx4 = w0*(1)**3 + w1*(1)**2 - w2*(1)
plt.plot([-1, 1], [ytestx0, ytestx4], 'b')

plt.title('initial fit')
plt.show()

w0record, w1record, w2record = [], [], []

# Training loop
for i in range(iteration):
    grad_w0 = 0
    grad_w1 = 0
    grad_w2 = 0
    cost = 0
    for j in range(len(x)):
        
        w0 -= eta * grad_w0
        w1 -= eta * grad_w1
        w2 -= eta * grad_w2
        
        w0record.append(w0)
        w1record.append(w1)
        w2record.append(w2)
        grad_w0 = -(d[j] - w0 * x[j] ** 3 - w1 * x[j] ** 2 - w2 * x[j]) * x[j]* x[j] * x[j]
        grad_w1 = -(d[j] - w0 * x[j] ** 3 - w1 * x[j] ** 2 - w2 * x[j]) * x[j] * x[j]
        grad_w2 = -(d[j] - w0 * x[j] ** 3 - w1 * x[j] ** 2 - w2 * x[j]) * x[j] 
        cost = cost + 0.5*(d[j] - w0*x[j]**3 - w1*x[j]**2 - w2*x[j])**2

# Plotting the results
ytestx0 = w0 * (-1) ** 3 + w1 * (-1) ** 2 - w2 * (-1)
ytestx4 = w0 * 1 ** 3 + w1 * 1 ** 2 - w2 * 1

plt.figure()
plt.plot([-1, 1], [ytestx0, ytestx4], 'b')
plt.title('optimal fit')

plt.figure()
plt.plot(w0record)
plt.title('w0')

plt.figure()
plt.plot(w1record)
plt.title('w1')

plt.figure()
plt.plot(w2record)
plt.title('w2')

plt.show()


