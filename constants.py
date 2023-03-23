import numpy as np
import math
from math import sqrt


# Crazyflie mass and inertia

m = 0.027
J = (10**-6) * np.array([[16.57, 0.83, 0.72],
              [0.83, 16.65, 1.8],
              [0.72, 1.8, 29.26]])


g = 9.81S
l = 0.0326695
d = sqrt(2) * l
c_tauf = 0.005964552

# mixing matrix

mat = np.array([[1, 1, 1, 1], [0, d, 0, -d], [-d, 0, d, 0], [-c_tauf, c_tauf, -c_tauf, c_tauf]])
mat_inv = np.linalg.inv(mat)
a0 = 4 * 1.563383 * (10**-5)

# control gains

kx = 0.8
kv = 0.4
kR = 0.001
kOm = 0.000065

Kx = np.diag([0.08, 0.08, 0.12])
Kv = np.diag([0.06, 0.06, 0.07])
KR = np.diag([0.00065, 0.00065, 0.0007])
KOm = np.diag([0.00004, 0.00004, 0.00005])

