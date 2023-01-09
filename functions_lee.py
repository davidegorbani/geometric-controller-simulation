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
    output = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return output

def vee_map(x):
    output = np.array([x[2,1], x[0,2], x[1,0]])
    return output

def err_Om(R, R_d, Om, Om_d):
    prod1 = np.dot(np.transpose(R),R_d)
    output = Om - np.dot(prod1,Om_d)
    return output


def lee_geom_cont(z, t):

    x = np.array([z[0], z[1], z[2]])
    v = np.array([z[3], z[4], z[5]])
    R = np.reshape(z[6:15],(3,3))
    Om = np.array([z[15], z[16],z[17]])

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
    f_vect = - kx*ex - kv*ev + m*g*e3 + m * x_2d_des
    f =  np.inner(f_vect, b3)
    ev_dot = - g * e3 + f / m * b3 - x_2d_des
    f_vect_dot = - kx * ev - kv * ev_dot + m * x_3d_des
    f_dot = np.inner(f_vect_dot, b3) + np.inner(f_vect, b3_dot)
    ev_ddot = f_dot / m * b3 + f / m * b3_dot - x_3d_des
    f_vect_ddot = - kx * ev_dot - kv * ev_ddot + m * x_4d_des

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

    e_R = 0.5 * vee_map(np.dot(np.transpose(R_des),R) - np.dot(np.transpose(R),R_des))
    e_Om = err_Om(R, R_des, Om, Om_des)
    
    M = -kR * e_R - kOm * e_Om - np.linalg.multi_dot([J, Om_hat, np.transpose(R), R_des, Om_des]) + np.linalg.multi_dot([J, np.transpose(R), R_des, Om_des_d])

    return f, M