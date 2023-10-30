import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-1, 1, 0.05)
l = len(x)
d = 0.8* x**3 + 0.3 * x**2 - 0.4*x + np.random.normal(0, 0.02, l)

plt.plot(x, d)
plt.show()

