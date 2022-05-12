# Estimating Hubble Constant using Linear Regression Method

# Linear Regression is a Machine Learning method. We use both mean absolute error (error_1) function and also
# mean squared error function (error_2) and we use Gradient decent algorithm to compute minimum of the cost function.

import numpy as np
import matplotlib.pyplot as plt

file1 = open('sn.txt', 'r')
Lines = file1.readlines()

z = []
distance_mod = []

for line in Lines:
    x = line.split()
    if float(x[1]) < 0.1:  # data upto redshift 0.1
        z.append(float(x[1]))
        distance_mod.append(float(x[2]))

# Calculating Distance from Hubble Lemaitre Law with H_0 = 71
z = np.array(z)
distance_mod = np.array(distance_mod)

c = 3 * 1e5
distance = 71 * distance_mod / c

m = 10
n = 0.008
error_1 = 0.1
error_2 = 0.1
a = 0.8

while error_1 > 0.0000005:
    h = m * z + n
    m = m - a * (1 / len(z)) * np.sum((h - distance) * z)
    n = n - a * (1 / len(z)) * np.sum(h - distance)
    error_1 = np.abs(np.sum(h - distance))  # mean absolute error
    error_2 = np.sum((h - distance) ** 2) / len(z)  # mean squared error
print('Hubble Constant H_0 =', 1 / m)

plt.scatter(z, distance)
plt.plot((min(z), max(z)), (m * min(z) + n, m * max(z) + n), 'r-')
plt.show()
