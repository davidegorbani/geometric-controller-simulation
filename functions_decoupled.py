import numpy as np
from math import pi
from constants import *

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

def exp_map(theta, v):
    output = np.identity(3) + np.sin(theta) * hat_map(v) + (1 - np.cos(theta)) * np.dot(hat_map(v), hat_map(v))
    return output

def err_Om(R, R_d, Om, Om_d):
    prod1 = np.dot(np.transpose(R),R_d)
    output = Om - np.dot(prod1,Om_d)
    return output


def decoupled_cont(x, v, R, Om, t):

    e3 = np.array([0, 0, 1])
    Om_hat = hat_map(Om)
    b3 = np.dot(R,e3)
    b3_dot = np.linalg.multi_dot([R, Om_hat, e3])

    x_des = np.array([0.4*t, 0.4*np.sin(pi*t), 0.6*np.cos(pi*t)])
    x_d_des = np.array([0.4, 0.4*pi*np.cos(pi*t), -0.6*pi*np.sin(pi*t)])
    x_2d_des = np.array([0, -0.4*pi**2*np.sin(pi*t), -0.6*pi**2*np.cos(pi*t)])
    x_3d_des = np.array([0, -0.4*(pi**3)*np.cos(pi*t), 0.6*(pi**3)*np.sin(pi*t)])
    x_4d_des = np.array([0, 0.4*(pi**4)*np.sin(pi*t), 0.6*(pi**4)*np.cos(pi*t)])
    b1_dir = np.array([np.cos(pi*t), np.sin(pi*t), 0])
    b1_dir_dot = pi * np.array([-np.sin(pi*t), np.cos(pi*t), 0])
    b1_dir_ddot = (pi**2) * np.array([-np.cos(pi*t), -np.sin(pi*t), 0])

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
    f_vect = - kx*ex - kv*ev - m*g*e3 + m * x_2d_des
    f = - np.inner(f_vect, b3)
    ev_dot = g * e3 - f / m * b3 - x_2d_des
    f_vect_dot = - kx * ev - kv * ev_dot + m * x_3d_des
    f_dot = -np.inner(f_vect_dot, b3) - np.inner(f_vect, b3_dot)
    ev_ddot = - f_dot / m * b3 - f / m * b3_dot - x_3d_des
    f_vect_ddot = - kx * ev_dot - kv * ev_ddot + m * x_4d_des

    b3_des, b3_des_dot, b3_des_ddot = deriv_vect(-f_vect, -f_vect_dot, -f_vect_ddot)

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
    
    M = decoupled_attitude(Om, R, R_des, Om_des, Om_des_d, b1_des, b3_des, b3_des_dot)

    return f, M

def decoupled_attitude(Om, R, R_des, Om_des, Om_des_dot, b1_des, b3_des, b3_des_dot):
    kb = 10
    kom = 7
    ky = 5
    kom_y = 8
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    b1 = np.dot(R, e1)
    b2 = np.dot(R, e2)
    b3 = np.dot(R, e3)
    b3_hat = hat_map(b3)
    J1 = J[0, 0]
    J3 = J[2, 2]
    
    Om12 = Om[0] * b1 + Om[1] * b2
    Om12_des = np.cross(b3_des, b3_des_dot)
    b3_dot = np.cross(Om12, b3)

    eb = np.cross(b3_des, b3)
    eom = Om12 + np.linalg.multi_dot((b3_hat, b3_hat, Om12_des))

    tau = -kb * eb - kom * eom - J1 * np.inner(b3, Om12_des) * b3_dot - J1 * np.linalg.multi_dot((b3_hat, b3_hat, Om12_des))
    M1 = np.inner(tau, b1) + J3 * Om[1] * Om[2]
    M2 = np.inner(tau, b2) - J3 * Om[2] * Om[0]

    ey = -np.inner(b1_des, b2)
    eom_y = Om[2] - Om_des[2]
    M3 = - ky * ey - kom_y * eom_y + J3 * Om_des_dot[2]

    return np.array([M1, M2, M3])


