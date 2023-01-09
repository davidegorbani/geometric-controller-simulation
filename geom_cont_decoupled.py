from math import pi
import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from constants import g, m, J
from functions_decoupled import decoupled_cont, hat_map
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def model(z, t):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    J_inv = inv(J)
    f, M = decoupled_cont(x, v, R, Om, t)
    
    dxdt = v
    dvdt = g * e3 - f * b3 / m
    dRdt = np.dot(R, Om_hat)
    dOmdt = np.dot(J_inv, (M - np.cross(Om, np.dot(J, Om))))
    dzdt = np.concatenate((dxdt, dvdt, dRdt.flatten(), dOmdt))

    return dzdt


x0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])
R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Om0 = np.array([0, 0, 0])

z0 = np.concatenate((x0, v0, R0, Om0), axis=None)

n = 1001
tf = 10.0
t = np.linspace(0,tf,n)


sol = odeint(model, z0, t)

x = sol[:,0]
y = sol[:,1]
z = sol[:,2]

x_des = 0.4 * t
y_des = 0.4 * np.sin(pi*t)
z_des = 0.6 * np.cos(pi*t)

fig1 = plt.figure()
plt.plot(t,x)
plt.plot(t,x_des)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("x [m]")
plt.show()

fig2 = plt.figure()
plt.plot(t,y)
plt.plot(t,y_des)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("y [m]")
plt.show()

fig3 = plt.figure()
plt.plot(t,z)
plt.plot(t,z_des)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("z [m]")
plt.show()

ax = plt.axes(projection='3d')
ax.plot(x, y, z)
plt.show()

