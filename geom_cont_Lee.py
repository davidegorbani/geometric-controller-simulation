import numpy as np
from math import pi
from functions_lee import lee_geom_cont, hat_map
from numpy.linalg import inv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from constants import J, g, m
from scipy.spatial.transform import Rotation

def model(z, t, f, M):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    J_inv = inv(J)
    
    dxdt = v
    dvdt = - g * e3 + f * b3 / m
    dRdt = np.dot(R, Om_hat)
    dOmdt = np.dot(J_inv, M)
    dzdt = np.concatenate((dxdt, dvdt, dRdt.flatten(), dOmdt))

    return dzdt

rpy = np.array([0, 0, 0])

x0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])
Om0 = np.array([0, 0, 0])
r = Rotation.from_euler('xyz', rpy, degrees=True)
rot = r.as_matrix()
R0 = np.reshape(rot, (3, 3)).T

z0 = np.concatenate((x0, v0, R0, Om0), axis=None)

n = 2001
tf = 10.0
t = np.linspace(0, tf, n)
x = np.empty_like(t)
y = np.empty_like(t)
z = np.empty_like(t)
mat_rot = np.zeros((9, n))

for i in range(1, n):

    f, M = lee_geom_cont(z0, t[i-1])
    tspan = [t[i-1], t[i]]
    sol = odeint(model, z0, tspan, args=(f, M,))
    z0 = sol[1]
    x[i] = z0[0]
    y[i] = z0[1]
    mat_rot[:, i - 1] = z0[6:15]
    z[i] = z0[2]

x_des = 0.4 * t
y_des = 0.4 * np.sin(pi * t)
z_des = 0.6 * np.cos(pi * t) 

fig1 = plt.figure()
plt.plot(t, x, label="sim")
plt.plot(t, x_des, label="ref")
plt.xlabel("time [s]")
plt.ylabel("x [m]")
plt.grid()
plt.legend()
plt.show()

fig2 = plt.figure()
plt.plot(t, y, label="sim")
plt.plot(t, y_des, label="ref")
plt.xlabel("time [s]")
plt.ylabel("y [m]")
plt.grid()
plt.legend()
plt.show()

fig3 = plt.figure()
plt.plot(t, z, label="sim")
plt.plot(t, z_des, label="ref")
plt.xlabel("time [s]")
plt.ylabel("z [m]")
plt.grid()
plt.legend()
plt.show()

roll = []
pitch = []
yaw = []
mat_rot = mat_rot.T

for jj in range(0, n):
    mat = [[mat_rot[jj, 0], mat_rot[jj, 1], mat_rot[jj, 2]], [mat_rot[jj, 3], mat_rot[jj, 4], mat_rot[jj, 5]], [mat_rot[jj, 6], mat_rot[jj, 7], mat_rot[jj, 8]]]
    r = Rotation.from_matrix(mat)
    rpy = r.as_euler('xyz', degrees=True)
    roll.append(rpy[0])
    pitch.append(rpy[1])
    yaw.append(rpy[2])

fig4 = plt.figure()
plt.plot(t, roll)
plt.xlabel("time [s]")
plt.ylabel("roll [deg]")
plt.grid()
plt.show()

fig5 = plt.figure()
plt.plot(t, pitch)
plt.xlabel("time [s]")
plt.ylabel("pitch [deg]")
plt.grid()
plt.show()

fig5 = plt.figure()
plt.plot(t, yaw)
plt.xlabel("time [s]")
plt.ylabel("yaw [deg]")
plt.grid()
plt.show()


