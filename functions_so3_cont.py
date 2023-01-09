import numpy as np
from math import pi
import matplotlib.pyplot as plt
from math import sqrt

g = 9.81
l = 0.0326695
d = sqrt(2) * l
c_tauf = 0.005964552
# m = 1
# J = np.diag([0.07, 0.07, 0.12])

m = 0.027
J = (10**-6) * np.array([[16.57, 0.83, 0.72],
              [0.83, 16.65, 1.8],
              [0.72, 1.8, 29.26]])

mat = np.array([[1, 1, 1, 1], [0, d, 0, -d], [-d, 0, d, 0], [-c_tauf, c_tauf, -c_tauf, c_tauf]])
mat_inv = np.linalg.inv(mat)
a0 = 4 * 1.563383 * (10**-5)

kx = np.diag([0.15, 0.15, 0.25])
kv = np.diag([0.06, 0.06, 0.07])
ki = np.diag([0.006, 0.006, 0.007])
k_rot = np.diag([0.003, 0.003, 0.005]) * 2
k_ang_vel = np.diag([0.00003, 0.00003, 0.00006]) * 5

# kx = np.diag([0.1, 0.1, 0.12])
# kv = np.diag([0.08, 0.08, 0.09])
# ki = np.diag([0.0, 0.0, 0.0])
# k_rot = np.diag([0.0005, 0.0005, 0.0007])
# k_ang_vel = np.diag([0.00003, 0.00003, 0.00004])

We = np.diag([2, 2, 2, 1, 1, 1, 0.001, 0.001, 0.1])
Wf = np.diag([0.1, 0.1, 1])

def deriv_vect(a, a_d, a_dd):
    a_norm = np.linalg.norm(a)
    b = a / a_norm
    b_dot = a_d / a_norm - a * np.inner(a, a_d) / a_norm**3
    b_ddot = a_dd / a_norm - a_d / (a_norm**3) * (2 * np.inner(a, a_d)) - a / (a_norm**3) *(np.inner(a_d, a_d) + np.inner(a, a_dd)) + 3 * a / (a_norm**5) * (np.inner(a, a_d)**2)

    return b, b_dot, b_ddot

def hat_map(x):
    output = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]) # type: ignore
    return output

def vee_map(x):
    output = np.array([x[2,1], x[0,2], x[1,0]])
    return output

def Log(R):
    a = (np.matrix.trace(R)-1)/2
    phi = np.arccos(bound(a, -1, 1))
    den = (2 * np.sinc(phi / 2 / pi) * np.cos(phi / 2))
    
    if np.abs(den) < 1e-5:
        w, v = np.linalg.eig(np.around(R, decimals=8))
        ind = np.argmax(np.real(w))
        u = np.array([v[0, ind], v[1, ind], v[2, ind]])
        log = phi * np.real(u) / np.linalg.norm(u)
    else:
        log = vee_map(R - np.transpose(R)) / den
    
    return log

def inv_left_Jacobian(r):
    phi = np.linalg.norm(r)
    I = np.identity(3)
    if phi < 1e-10:
        output = I
    else:
        u = r / phi
        output = I - phi / 2 * hat_map(u) + (1 - np.cos(phi / 2) / np.sinc(phi / 2 / pi)) * np.dot(hat_map(u), hat_map(u))
    
    return output

def bound(a, min, max):
    
    if a > max:
        out = max
    elif a < min:
        out = min
    else:
        out = a
    
    return out

def controller_so3(z, t):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])
    eI1 = bound(z[18], -1, 1)
    eI2 = bound(z[19], -1, 1)
    eI3 = bound(z[20], -1, 1)
    eI = np.array([eI1, eI2, eI3])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    b3_dot = np.linalg.multi_dot([R, Om_hat, e3])

    # r_y = 0.8
    # r_z = 0.4
    # om = 4
    # x_des = np.array([0.5*t, r_y*np.sin(om*t), r_z*np.cos(om*t)])
    # x_d_des = np.array([0.5, r_y*om*np.cos(om*t), -r_z*om*np.sin(om*t)])
    # x_2d_des = np.array([0, -r_y*om**2*np.sin(om*t), -r_z*om**2*np.cos(om*t)])
    # x_3d_des = np.array([0, -r_y*(om**3)*np.cos(om*t), r_z*(om**3)*np.sin(om*t)])
    # x_4d_des = np.array([0, r_y*(om**4)*np.sin(om*t), r_z*(om**4)*np.cos(om*t)])
    # b1_dir = np.array([np.cos(om*t), np.sin(om*t), 0])
    # b1_dir_dot = om * np.array([-np.sin(om*t), np.cos(om*t), 0])
    # b1_dir_ddot = (om**2) * np.array([-np.cos(om*t), -np.sin(om*t), 0])

    # pos_x = (1 - np.exp(-t)) + 0.3 * t
    # pos_y = -0.2 * np.exp(-t) + 0.2 * np.cos(t)
    # pos_z = (1 - np.exp(-t)) + 0.4 * np.sin(t)
    # x_des = np.array([pos_x, pos_y, pos_z])
    # x_d_des = np.array([np.exp(-t), np.exp(-t), np.exp(-t)]) + np.array([0.3, -0.2 * np.sin(t), 0.4 * np.cos(t)])
    # x_2d_des = np.array([-np.exp(-t), -np.exp(-t), -np.exp(-t)]) + np.array([0, -0.2 * np.cos(t), - 0.4 * np.sin(t)])
    # x_3d_des = np.array([np.exp(-t), np.exp(-t), np.exp(-t)]) + np.array([0 , 0.2 * np.sin(t), -0.4 * np.cos(t)])
    # x_4d_des = np.array([-np.exp(-t), -np.exp(-t), -np.exp(-t)]) + np.array([0, 0.2 * np.cos(t), 0.4 * np.sin(t)])
    # b1_dir = np.array([np.cos(t), np.sin(t), 0])
    # b1_dir_dot = np.array([-np.sin(t), np.cos(t), 0])
    # b1_dir_ddot = np.array([-np.cos(t), -np.sin(t), 0])

    # x_des = np.array([5*np.cos((pi / 2.5 * 2)*t), 5*np.sin((pi / 2.5 * 2)*t), -5])
    # x_d_des = np.array([-5*(pi / 2.5 * 2)*np.sin((pi / 2.5 * 2)*t), 5*(pi / 2.5 * 2)*np.cos((pi / 2.5 * 2)*t), 0])
    # x_2d_des = np.array([-5*(pi / 2.5 * 2)**2*np.cos((pi / 2.5 * 2)*t), -5*(pi / 2.5 * 2)**2*np.sin((pi / 2.5 * 2)*t), 0])
    # x_3d_des = np.array([5*(pi / 2.5 * 2)**3*np.sin((pi / 2.5 * 2)*t), -5*((pi / 2.5 * 2)**3)*np.cos((pi / 2.5 * 2)*t), 0])
    # x_4d_des = np.array([5*(pi / 2.5 * 2)**4*np.cos((pi / 2.5 * 2)*t), 5*((pi / 2.5 * 2)**4)*np.sin((pi / 2.5 * 2)*t), 0])
    # b1_dir = np.array([np.cos((pi / 2.5)*t), np.sin((pi / 2.5)*t), 0])
    # b1_dir_dot = (pi / 2.5) * np.array([-np.sin((pi / 2.5)*t), np.cos((pi / 2.5)*t), 0])
    # b1_dir_ddot = ((pi / 2.5)**2) * np.array([-np.cos((pi / 2.5)*t), -np.sin((pi / 2.5)*t), 0])

    r = 2
    om = pi
    x_des = np.array([r*np.cos(om*t), r*np.sin(om*t), 1])
    x_d_des = np.array([-r*om*np.sin(om*t), r*om*np.cos(om*t), 0])
    x_2d_des = np.array([-r*om**2*np.cos(om*t), -r*om**2*np.sin(om*t), 0])
    x_3d_des = np.array([r*om**3*np.sin(om*t), -r*(om**3)*np.cos(om*t), 0])
    x_4d_des = np.array([r*om**4*np.cos(om*t), r*(om**4)*np.sin(om*t), 0])
    b1_dir = np.array([-np.sin(om*t), np.cos(om*t), 0])
    b1_dir_dot = np.array([-np.cos(om*t), -np.sin(om*t), 0])
    b1_dir_ddot = np.array([np.sin(om*t), -np.cos(om*t), 0])

    # x_des = np.array([0, 0, 0])
    # x_d_des = np.array([0, 0, 0])
    # x_2d_des = np.array([0, 0, 0])
    # x_3d_des = np.array([0, 0, 0])
    # x_4d_des = np.array([0, 0, 0])
    # b1_dir = np.array([1, 0, 0])
    # b1_dir_dot = np.array([0, 0, 0])
    # b1_dir_ddot = np.array([0, 0, 0])

    ex = x - x_des
    ev = v - x_d_des
    
    f_vect = (- np.dot(kx, ex) - np.dot(kv, ev) - np.dot(ki, eI) + m * g * e3 + m * x_2d_des)
    f =  np.inner(f_vect, b3)
    ev_dot = - g * e3 + f / m * b3 - x_2d_des
    f_vect_dot = (- np.dot(kx, ev) - np.dot(kv, ev_dot) - np.dot(ki, ex) + m * x_3d_des)
    f_dot = np.inner(f_vect_dot, b3) + np.inner(f_vect, b3_dot)
    ev_ddot = f_dot / m * b3 + f / m * b3_dot - x_3d_des
    f_vect_ddot = (- np.dot(kx, ev_dot) - np.dot(kv, ev_ddot) - np.dot(ki, ev) + m * x_4d_des)

    b3_des, b3_des_dot, b3_des_ddot = deriv_vect(f_vect, f_vect_dot, f_vect_ddot)

    A2 = - np.cross(b1_dir, b3_des)
    A2_dot = - np.cross((b1_dir_dot), b3_des) - np.cross((b1_dir), b3_des_dot)
    A2_ddot = - np.cross((b1_dir_ddot), b3_des) - 2 * np.cross((b1_dir_dot), b3_des_dot) - np.cross((b1_dir), b3_des_ddot)

    b2_des, b2_des_dot, b2_des_ddot = deriv_vect(A2, A2_dot, A2_ddot)

    b1_des = np.cross(b2_des, b3_des)
    b1_des_dot = np.cross(b2_des_dot, b3_des) + np.cross(b2_des, b3_des_dot)
    b1_des_ddot = np.cross(b2_des_ddot, b3_des) + 2 * np.cross(b2_des_dot, b3_des_dot) + np.cross(b2_des, b3_des_ddot)

    R_des = np.array([b1_des, b2_des, b3_des]).T
    R_des_d = np.array([b1_des_dot, b2_des_dot, b3_des_dot]).T
    R_des_ddot = np.array([b1_des_ddot, b2_des_ddot, b3_des_ddot]).T

    Om_des = vee_map(np.dot(np.transpose(R_des),R_des_d))
    Om_des_d = vee_map(np.dot(np.transpose(R_des),R_des_ddot) - np.dot(hat_map(Om_des), hat_map(Om_des)))

    R_db = np.dot(np.transpose(R), R_des)
    # print(R_db)
    rot_err = Log(R_db)
    # print(rot_err)
    ang_vel_err = np.dot(R_db, Om_des) - Om

    M = np.linalg.multi_dot((inv_left_Jacobian(rot_err).T, k_rot, rot_err)) + np.dot(k_ang_vel, ang_vel_err) + np.linalg.multi_dot((J, R_db, Om_des_d)) - np.linalg.multi_dot((J, hat_map(Om), R_db, Om_des))

    return f, M, ex, R_des

def controller_so3_attitude(z, t):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])
    e_r = np.array([1, 0, 0])
    eI = np.array([0, 0, 0])

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    b3_dot = np.linalg.multi_dot([R, Om_hat, e3])

    R_des = np.identity(3) + np.sin(4 * pi * t) * hat_map(e_r) + (1 - np.cos(4 * pi * t)) * (np.dot(e_r, e_r.T) - np.identity(3))
    Om_des = 4 * pi * e_r
    Om_des_d = np.array([0, 0, 0])


    f_vect = m * g * e3
    f =  np.inner(f_vect, b3)



    R_db = np.dot(np.transpose(R), R_des)
    # print(R_db)
    rot_err = Log(R_db)
    # print(rot_err)
    ang_vel_err = np.dot(R_db, Om_des) - Om

    M = np.linalg.multi_dot((inv_left_Jacobian(rot_err).T, k_rot, rot_err)) + np.dot(k_ang_vel, ang_vel_err) + np.linalg.multi_dot((J, R_db, Om_des_d)) - np.linalg.multi_dot((J, hat_map(Om), R_db, Om_des))
    
    T = np.array([f, M[0], M[1], (M[2])])
    thrust = np.dot(mat_inv, T)
    i = 0
    for t in thrust:
        if t < 0.000548456:
            thrust[i] = 0.000548456
        elif t > 0.1597:
            thrust[i] = 0.1597
        i = i + 1
    
    force = np.dot(mat, thrust)
    a = force[0]
    b = np.array([force[1], force[2], force[3]])

    return a, b, thrust

