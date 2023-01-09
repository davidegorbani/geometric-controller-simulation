import numpy as np
from math import pi
from numpy.linalg import inv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functions_so3_cont import g, m, J, hat_map, controller_so3
from scipy.spatial.transform import Rotation

# quadrotor model

def model(z, t, ex, f, M):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16], z[17]])
    eI = np.array([z[18], z[19], z[20]])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    J_inv = inv(J)

    dxdt = v
    dvdt = - g * e3 + f * b3 / m
    dRdt = np.dot(R, Om_hat)
    dOmdt = np.dot(J_inv, M)
    deIdt = ex
    dzdt = np.concatenate((dxdt, dvdt, dRdt.flatten(), dOmdt, deIdt))

    return dzdt


# initial conditions

x0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])
Om0 = np.array([0, 0, 0])
rpy = np.array([0, 0, 0])
r = Rotation.from_euler('xyz', rpy, degrees=True)
rot = r.as_matrix()
R0 = np.reshape(rot, (3, 3)).T
eI0 = np.array([0, 0, 0])

z0 = np.concatenate((x0, v0, R0, Om0, eI0), axis=None)

n = 2001
tf = 10.0
t = np.linspace(0, tf, n)
x = np.empty_like(t)
y = np.empty_like(t)
z = np.empty_like(t)
Om_x = np.empty_like(t)
Om_y = np.empty_like(t)
Om_z = np.empty_like(t)
Om_x_des = np.empty_like(t)
Om_y_des = np.empty_like(t)
Om_z_des = np.empty_like(t)
mat_rot = np.zeros((9, n))
mat_rot_des = np.zeros((9, n))

mat_rot[:, 0] = R0.reshape(9)

# simulation

for i in range(1, n):
    t_span = [t[i-1], t[i]]
    f, M, ex, R_des = controller_so3(z0, t[i-1])
    z1 = odeint(model, z0, t_span, args=(ex, f, M,))

    z0 = z1[1]
    x[i] = z0[0]
    y[i] = z0[1]
    z[i] = z0[2]
    mat_rot[:, i] = z0[6:15]
    mat_rot_des[:, i] = R_des.reshape(9)


# desired trajectory

r = 2
om = pi
x_des = r*np.cos(om*t)
y_des = r*np.sin(om*t)
z_des = 1 + 0 * t

# comparison between the simulated and desired trajecotries

fig1 = plt.figure()
plt.plot(t,x, label='sim')
plt.plot(t,x_des, label='des')
plt.ylabel('x(t) [m]')
plt.xlabel('time [s]')
plt.legend()
plt.grid()
plt.show()

fig2 = plt.figure()
plt.plot(t,y, label='sim')
plt.plot(t,y_des, label='des')
plt.ylabel('y(t) [m]')
plt.xlabel('time [s]')
plt.legend()
plt.grid()
plt.show()

fig3 = plt.figure()
plt.plot(t,z, label='sim')
plt.plot(t,z_des, label='des')
plt.ylabel('z(t) [m]')
plt.xlabel('time [s]')
plt.legend()
plt.grid()
plt.show()

# mapping the attitude from rotation matrix to Euler angles

roll = []
pitch = []
yaw = []
mat_rot = mat_rot.T
mat_rot_des = mat_rot_des.T
roll_d = []
pitch_d = []
yaw_d = []

for jj in range(0, n):
    mat = [[mat_rot[jj, 0], mat_rot[jj, 1], mat_rot[jj, 2]], [mat_rot[jj, 3], mat_rot[jj, 4], mat_rot[jj, 5]], [mat_rot[jj, 6], mat_rot[jj, 7], mat_rot[jj, 8]]]
    r = Rotation.from_matrix(mat)
    rpy = r.as_euler('xyz', degrees=True)
    roll.append(rpy[0])
    pitch.append(rpy[1])
    yaw.append(rpy[2])

for jj in range(0, n):
    mat_d = [[mat_rot_des[jj, 0], mat_rot_des[jj, 1], mat_rot_des[jj, 2]], [mat_rot_des[jj, 3], mat_rot_des[jj, 4], mat_rot_des[jj, 5]], [mat_rot_des[jj, 6], mat_rot_des[jj, 7], mat_rot_des[jj, 8]]]
    r = Rotation.from_matrix(mat_d)
    rpy = r.as_euler('xyz', degrees=True)
    roll_d.append(rpy[0])
    pitch_d.append(rpy[1])
    yaw_d.append(rpy[2])

# plots of the desired and simulated roll, pitch and yaw angles

fig4 = plt.figure()
plt.plot(t, roll)
plt.plot(t, roll_d)
plt.grid()
plt.ylabel('roll(t) [deg]')
plt.xlabel('time [s]')
plt.show()

fig5 = plt.figure()
plt.plot(t, pitch)
plt.plot(t, pitch_d)
plt.grid()
plt.ylabel('pitch(t) [deg]')
plt.xlabel('time [s]')
plt.show()

fig5 = plt.figure()
plt.plot(t, yaw)
plt.plot(t, yaw_d)
plt.grid()
plt.ylabel('yaw(t) [deg]')
plt.xlabel('time [s]')
plt.show()
