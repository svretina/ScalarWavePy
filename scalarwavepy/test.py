#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def constant(x, c):
    return np.full(x.shape, c)


def linear(x, slope):
    return slope * x


def quadratic(x):
    return x * x


x = np.linspace(-10, 10, 21)
# print(x)
y1 = constant(x, 5)
y2 = linear(x, 3)
y3 = quadratic(x)

y1d = np.gradient(y1)
y2d = np.gradient(y2)
y3d = np.gradient(y3)
# print(y3)
print(y3d)
print(2 * x)
exit()
plt.plot(x, y1)
plt.plot(x, y1d)
plt.show()
plt.clf()

plt.plot(x, y2)
plt.plot(x, y2d)
plt.show()
plt.clf()

plt.plot(x, y3)
plt.plot(x, y3d, "-x")
plt.plot(x, 2 * x, "-o")
plt.show()
plt.clf()
