#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

""" Formats date from dataset """
def bytespdate2num(fmt, encoding='utf-8'):
    def bytesconverter(b):
        s = b.decode(encoding)
        return (mdates.datestr2num(s))
    return bytesconverter


""" Plots the results from the EKF

    Outputs:
     - Result of the EKF algorithm
     - x-axis -> x position
     - y-axis -> y position
     - normal line -> EKF estimate
     - '+' -> GPS measurements
     - arrows -> estimated heading
"""
def plot():
    print("Plotting graph...")
    fig = plt.figure(figsize=(16,9))  # change this if you need to

    # plots the x position, y position and heading
    plt.quiver(x_pos, y_pos, np.cos(head), np.sin(head), color='#FFFF00', units='xy', width=0.05, scale=0.5)
    
    # plots estimated state from the EKF
    plt.plot(x_pos, y_pos, label='Estimated Position from EKF', c='#000000', lw=2)

    # plots each measurement from GPS sensor    
    plt.scatter(change_x[::5], change_y[::5], s=50, label='GPS Measurements', marker='+')

    # plot start and finish (initial pos and final pos)
    plt.scatter(x_pos[0], y_pos[0], s=60, label='First Measurement', c='#006400')
    plt.scatter(x_pos[-1], y_pos[-1], s=60, label='Last Measurement', c='#FF0000')
    
    # graph format
    plt.xlabel('X Position (metres)')
    plt.ylabel('Y Position (metres)')
    plt.title('Extended Kalman Filter')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()


""" Estimates the next value for the state vector

    Inputs:
     - X_hat_t_1 (the previous measurement)
     - P_t_1 (the previous measurement's covariance matrix)
     - Q_t (the measurement noise matrix)
    
    Outputs:
     - X_hat_t (the predicted value of the state estimate)
     - P_t_1 (the predicted state estimate's covariance matrix)
"""
def prediction(X_hat_t_1, P_t_1, Q_t):
    X_hat = X_hat_t_1

    dt = 1.0 / 50.0

    X_hat_t[0] = X_hat_t_1[0] + X_hat_t_1[3]*dt * np.cos(X_hat_t_1[2])
    X_hat_t[1] = X_hat_t_1[1] + X_hat_t_1[3]*dt * np.sin(X_hat_t_1[2])
    X_hat_t[2] = X_hat_t_1[2]
    X_hat_t[3] = X_hat_t_1[3] + X_hat_t_1[5]*dt
    X_hat_t[4] = 0.0000001
    X_hat_t[5] = X_hat_t_1[5]

    J_13 = float((X_hat_t[3]/X_hat_t[4]) * (np.cos(X_hat_t[4]*dt+X_hat_t[2]) - np.cos(X_hat_t[2])))
    J_14 = float((1.0/X_hat_t[4]) * (np.sin(X_hat_t[4]*dt+X_hat_t[2]) - np.sin(X_hat_t[2])))
    J_15 = float((dt*X_hat_t[3]/X_hat_t[4])*np.cos(X_hat_t[4]*dt+X_hat_t[2]) - (X_hat_t[3]/X_hat_t[4]**2)*(np.sin(X_hat_t[4]*dt+X_hat_t[2]) - np.sin(X_hat_t[2])))
    J_23 = float((X_hat_t[3]/X_hat_t[4]) * (np.sin(X_hat_t[4]*dt+X_hat_t[2]) - np.sin(X_hat_t[2])))
    J_24 = float((1.0/X_hat_t[4]) * (-np.cos(X_hat_t[4]*dt+X_hat_t[2]) + np.cos(X_hat_t[2])))
    J_25 = float((dt*X_hat_t[3]/X_hat_t[4])*np.sin(X_hat_t[4]*dt+X_hat_t[2]) - (X_hat_t[3]/X_hat_t[4]**2)*(-np.cos(X_hat_t[4]*dt+X_hat_t[2]) + np.cos(X_hat_t[2])))

    # jacobian matrix of A
    J_A = np.matrix([[1.0, 0.0, J_13, J_14, J_15, 0.0],
                     [0.0, 1.0, J_23, J_24, J_25, 0.0],
                     [0.0, 0.0,  1.0,  0.0,   dt, 0.0],
                     [0.0, 0.0,  0.0,  1.0,  0.0,  dt],
                     [0.0, 0.0,  0.0,  0.0,  1.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0, 1.0]])
    
    # error covariance in measurements (6x1 matrix)
    P_t_1 = J_A * P_t_1 * J_A.T + Q_t

    return X_hat_t, P_t_1


""" Updates the prediction accounting for previous values

    Inputs:
     - X_hat_t (the current state estimate)
     - P_t (the current state estimate's covariance matrix)
     - Z_t (the conditioning matrix)
     - R_t (the measurement noise matrix)

    Outputs:
     - X_t (the new updated state estimate)
     - P_t (the new updated state estimate's covariance matrix)
"""
def update(X_hat_t, P_t, Z_t, R_t):
    hx = np.matrix([[float(X_hat_t[0])], [float(X_hat_t[1])], [float(X_hat_t[3])],
                    [float(X_hat_t[4])], [float(X_hat_t[5])]])
 
    J_H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])      
    
    S = J_H * P_t * J_H.T + R_t
    K = (P_t * J_H.T) * np.linalg.inv(S)

    # update the state vector estimate
    Z = Z_t.reshape(J_H.shape[0], 1)

    y = Z - hx
    X_t = X_hat_t + (K * y)
    
    # update the covariance matrices
    I = np.eye(X_hat_t.shape[0])
    P_t = (I - (K * J_H)) * P_t
 
    x_pos.append(float(X_t[0]))
    y_pos.append(float(X_t[1]))
    head.append(float(X_t[2]))

    return X_t, P_t


""" Runs the Extended Kalman Filter algorithm

    Makes an initial prediction from the initialised state vector.  It then updates
    this with all previous measurements.  Calls plot() to graphically represent the results.

    Inputs:
     - X_hat_t (the initial state vector)
     - P_t (the initial state vector's covariance matrix)
     - Q_t (the process noise matrix)
     - R_t (the measurement noise matrix)

    Outputs:
     - Calls plot() to make a graph of the results
"""
def extended_kalman_filter(X_hat_t, P_t, Q_t, R_t):
    for i in range(sensor_measurements.shape[1]):
        # calculate prediction of current state
        X_hat_t, P_hat_t = prediction(X_hat_t, P_t, Q_t)

        Z_t = sensor_measurements[:, i]   # take all previous measurements as Z_t

        # update the predictions and their covariances
        X_t, P_t = update(X_hat_t, P_hat_t, Z_t, R_t)

        X_hat_t = X_t    # updated predictions
        P_hat_t = P_t    # updated covariance matrices
    
    plot()


""" DISCLAIMER !!!! DO NOT TOUCH ANYTHING BELOW THIS OR I WILL FIND WHERE YOU LIVE !!!! """

# set path variables
PATH = "data.csv"

# check if the dataset file exists
# try:
#     PATH = sys.argv[1]
#     if not os.path.exists:
#         print("[!] Error: File does not exist [!]")
#         sys.exit(2)
# except (FileNotFoundError, IsADirectoryError):
#     print("[!] Error: Incorrect arguments passed [!]")
#     print("Usage: python ekf.py <path_to_dataset>")
#     sys.exit(2)

# read in the data from the .csv dataset file
date, time, millis, rollrate, pitchrate, yawrate, roll, pitch, yaw, speed, accel_x, accel_y, accel_z, heading, latitude, longitude, altitude = np.loadtxt(PATH, delimiter=',', unpack=True, 
                  converters={0: bytespdate2num('%Y-%m-%d')}, skiprows=1)

a = 2.0 * np.pi * (6378388.0 + altitude) / 360.0    # approx lat. and long. (converted to metres)

# calculate rate of change for x and y
change_x = np.cumsum(a * np.cos(latitude * np.pi / 180.0) * np.hstack((0.0, np.diff(longitude))))
change_y = np.cumsum(a * np.hstack((0.0, np.diff(latitude))))

# sensor measurements
sensor_measurements = np.vstack((change_x, change_y, speed/3.6, yawrate/180.0*np.pi, accel_x))

# initial state vector X_hat_1
X_hat_t = np.matrix([[change_x[0],                   # initial x pos (metres)
                      change_y[0],                   # initial y pos (metres)
                      heading[0] * np.pi / 180.0,    # initial heading (radians)
                      speed[0] / 3.6,                # initial speed (m/s)
                      yawrate[0] * np.pi / 180.0,    # initial yawrate (radians)
                      accel_x[0]]])                  # initial acceleration (m/s^-2) 
X_hat_t = X_hat_t.T  # transpose to make 6x1

# initial state covariance P_t
P_t = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

# process noise Q_t
Q_t = np.diag([(1.0/50.0*0.5*8.8)**2,    # noise in gps (x component)
               (1.0/50.0*0.5*8.8)**2,    # noise in gps (y component)
               (1.0/50.0*0.1)**2,        # noise in heading
               (1.0/50.0*8.8)**2,        # noise in speed
               (1.0/50.0)**2,            # noise in yawrate
                0.5**2])                 # noise in acceleration

# measurement noise R_t
R_t = np.diag([5.0**2,    # noise in gps (x component)
               5.0**2,    # noise in gps (y component)
               3.0**2,    # noise in speed
               0.1**2,    # noise in yawrate
               1.0**2])   # noise in acceleration

x_pos, y_pos, head = [], [], []

print("[+] UGRDV-21 - Velocity Estimation [+]\n")

print("Workflow:")
print("Running Extended Kalman Filter...")
print(f"Using {PATH.split('/')[-1]} as input data")

# run the algorithm
extended_kalman_filter(X_hat_t, P_t, Q_t, R_t)

print("\n[+] Execution terminated [+]")