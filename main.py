""" Extended Kalman Filter Implementation

    Currently runs from dataset as input:
    - uses the data from the sensors to calculate the predicted state vector
    - state vector includes 12 states:
            gps (x), gps (y), heading, speed (x), speed(y), yawrate, acceleration (x), acceleration(y), slip ratios (x4)
    - failure detection:
        > null errors
        > outlier values
        > drift values
    
    Output:
    - 12x1 array representing the state estimate
    - 12x6 array representing the state estimate's covariance (row per estimate)
"""
#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.dates as mdates

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

def calculate_tyre_loads(loads, accel, load_transfer, weight):
    """ Calculates lateral and longitudinal tyre loads 
    
        Inputs:
        - loads (loads of each tyre - initially 0)
        - accel (acceleration in either x/y direction)
        - load_transfer (longitudinal/latitudinal load transfer)
        - weight (weight of the car)

        Outputs:
        - loads (longitudinal/latitudinal tyre loads)
    """
    # acceleration +'ve => +'ve for rear wheels & -'ve for front wheels
    if accel > 0:
        for i in range(len(loads)):
            if (i % 2 == 0):
                loads[i] = (weight - load_transfer) / 2
            else:
                loads[i] = (weight + load_transfer) / 2
    else:
        for i in range(len(loads)):
            if (i % 2 == 0):
                loads[i] = (weight + load_transfer) / 2
            else:
                loads[i] = (weight - load_transfer) / 2
    
    return loads


def prediction(X_hat_t_1, P_t_1, Q_t):
    """ Estimates the next value for the state vector

        Inputs:
        - X_hat_t_1 (12x1 matrix: the previous measurement)
        - P_t_1 (12x6 matrix: the previous measurement's covariance matrix)
        - Q_t (12x1 matrix: the measurement noise matrix)
        
        Outputs:
        - X_hat_t (12x1 matrix:  the predicted value of the state estimate)
        - P_t_1 (12x6 matrix: the predicted state estimate's covariance matrix)
    """
    X_hat = X_hat_t_1

    dt = 1.0 / 50.0    # rate of change

    X_hat_t[0] = X_hat_t_1[0] + X_hat_t_1[3]*dt * np.cos(X_hat_t_1[2])
    X_hat_t[1] = X_hat_t_1[1] + X_hat_t_1[3]*dt * np.sin(X_hat_t_1[2])
    X_hat_t[2] = X_hat_t_1[2]
    X_hat_t[3] = X_hat_t_1[3] + X_hat_t_1[5]*dt
    X_hat_t[4] = X_hat_t_1[3] + X_hat_t_1[5]*dt
    X_hat_t[5] = 0.0000001
    X_hat_t[6] = X_hat_t_1[6]
    X_hat_t[7] = X_hat_t_1[7]
    X_hat_t[8] = X_hat_t_1[8]
    X_hat_t[9] = X_hat_t_1[9]
    X_hat_t[10] = X_hat_t_1[10]
    X_hat_t[11] = X_hat_t_1[11]

    # DO STUFF WITH THIS
    # Calculating the Jacobian, A, with respect to the state vector X_hat_t
    J_ab1 = X_hat_t[3]*np.cos(X_hat_t[2]) - (X_hat[4]*np.sin(X_hat_t[2]))
    J_ab2 = X_hat[4]*np.sin(X_hat_t[2]) + X_hat_t[3]*np.cos(X_hat_t[2])
    J_ab3 = (1/m)*(force_rear_x-force_front_y*np.sin(dt)+m*X_hat_t[4]*X_hat_t[5])
    J_ab4 = (1/m)*(force_rear_y+force_front_x*np.cos(dt)-m*X_hat_t[3]*X_hat_t[5])
    J_ab5 = (1/intertia_x)*(force_front_y*load_front*np.cos(dt)-force_rear_y*load_rear*something)

    # jacobian matrix of A (CHANGE NUMBERS)
    J_A = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    # error covariance in measurements (12x6 matrix)
    P_t_1 = J_A * P_t_1 * J_A.T + Q_t

    return X_hat_t, P_t_1


def update(X_hat_t, P_t, Z_t, R_t):
    """ Updates the prediction accounting for previous values

        Inputs:
        - X_hat_t (12x1 matrix: the current state estimate)
        - P_t (12x6 matrix: the current state estimate's covariance matrix)
        - Z_t (the conditioning matrix)
        - R_t (12x1 matrix: the measurement noise matrix)

        Outputs:
        - X_t (12x1 matrix: the new updated state estimate)
        - P_t (12x6 matrix: the new updated state estimate's covariance matrix)
    """
    hx = np.matrix([[float(X_hat_t[0])], [float(X_hat_t[1])], [float(X_hat_t[3])],
                    [float(X_hat_t[4])], [float(X_hat_t[5])], [float(X_hat_t[6])],
                    [float(X_hat_t[7])], [float(X_hat_t[8])], [float(X_hat_t[9])],
                    [float(X_hat_t[10])], [float(X_hat_t[11])])
 
    # change this (check maths obvs not correct)
    J_H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    # calculate kalman gain
    S = J_H * P_t * J_H.T + R_t
    K = (P_t * J_H.T) * np.linalg.inv(S)

    # update the state vector estimate
    Z = Z_t.reshape(J_H.shape[0], 1)

    y = Z - hx
    X_t = X_hat_t + (K * y)

    # update the covariance matrices
    I = np.eye(X_hat_t.shape[0])
    P_t = (I - (K * J_H)) * P_t
 
    # x_pos.append(float(X_t[0]))
    # y_pos.append(float(X_t[1]))
    # head.append(float(X_t[2]))

    return X_t, P_t


def extended_kalman_filter(X_hat_t, P_t, Q_t, R_t):
    """ Runs the Extended Kalman Filter algorithm

        Makes an initial prediction from the initialised state vector.  It then updates
        this with all previous measurements.  Calls plot() to graphically represent the results.

        Inputs:
        - X_hat_t (12x1 matrix: the initial state vector)
        - P_t (12x6 matrix: the initial state vector's covariance matrix)
        - Q_t (12x1 matrix: the process noise matrix)
        - R_t (12x1: the measurement noise matrix)

        Outputs:
        - X_t (12x1 matrix: the final state estimate)
        - P_t (12x6 matrix: the final state estimate's covariance matrix)
    """
    # CHANGE THIS FOR LIVE SYSTEM
    for i in range(sensor_measurements.shape[1]):
        # calculate prediction of current state
        X_hat_t, P_hat_t = prediction(X_hat_t, P_t, Q_t)

        # THIS WILL NEED TO BE CHANGED TO REFLECT LIVE SYSTEM
        Z_t = sensor_measurements[:, i]   # take all previous measurements as Z_t

        # update the predictions and their covariances
        X_t, P_t = update(X_hat_t, P_hat_t, Z_t, R_t)

        X_hat_t = X_t    # updated predictions
        P_hat_t = P_t    # updated covariance matrices
    
    return X_hat_t, P_hat_t


""" DISCLAIMER !!!! DO NOT TOUCH ANYTHING BELOW THIS OR I WILL FIND WHERE YOU LIVE !!!! """

#=================================================================================================================#

# Variables we need

fric_coefficient = 1.5
wheel_base = 1.53    # UGR's car wheel_base (m)
height_CoG = 0.26    # UGR's car w/ driver (m) (300mm w/ driver 250-260mm w/o driver)
mass = 201.5    # 201.5kg w/o driver and 227kg (2018 UGR)
car_weight = mass * 9.8

#=================================================================================================================#

# Longitudinal Tyre Stiffness #

load_transfer_long = car_weight * accel_x * height_CoG / wheel_base    # update step (probably)

front_left = front_right = rear_left = rear_right = 0.0    # loads on each tyre

tyre_loads = [front_left, rear_left, front_right, rear_right]

# tyre loads (longitudinal - x direction)
tyre_loads = calculate_tyre_loads(tyre_loads, accel_x, load_transfer_long, car_weight)

front_left_stiff = front_right_stiff = rear_left_stiff = rear_right_stiff = 0

longitudinal_tyre_stiffnesses = [front_left_stiff, rear_left_stiff, front_right_stiff, rear_right_stiff]

for i in range(len(longitudinal_tyre_stiffnesses)):
    longitudinal_tyre_stiffnesses[i] = fric_coefficient * tyre_loads[i] / 0.15

#=================================================================================================================#

# Lateral Tyre Stiffness #

track_width = 1.201    # IMECHE's DV car (m) - distance between rear wheels

load_transfer_lat = car_weight * accel_y * height_CoG / track_width

lateral_tyre_stiffnesses = [front_left_stiff, rear_left_stiff, front_right_stiff, rear_right_stiff]

# tyre loads (latitudinal - y direction)
tyre_loads = calculate_tyre_loads(tyre_loads, accel_y, load_transfer_lat, car_weight)

# formula given in degrees but all measurements in radians?????
for i in range(len(lateral_tyre_stiffnesses)):
    lateral_tyre_stiffnesses[i] = fric_coefficient * tyre_loads[i] / np.rad2deg(15*np.pi/180)

#=================================================================================================================#

# Yaw Moment of Inertia #

yaw_of_intertia = mass * (wheel_base**2 + track_width**2) / 12.0

#=================================================================================================================#

# F_m Forumla?? #

slip_angle = delta - arctan(vel_x / vel_y)

# denotes front wheels
F_x1 = longitudinal_tyre_stiffnesses * slip_ratio
F_x2 = F_x1    # not sure on this

# denotes rear wheels
F_y1 = lateral_tyre_stiffnesses * slip_angle
F_y2 = F_y1    # not sure on this

length_front = 1    # distance from front to centre
length_rear = 1     # distance from rear to centre

f_M = (1/yaw_of_intertia)*(length_front*(F_x1*np.sin(slip_angle)+F_y1*np.cos(slip_angle)-length_rear*F_y2))

#=================================================================================================================#

# Moment of Inertia of Wheel #

wheel_radius = height_CoG

intertia_of_wheel = 0.5 * wheel_radius**2 * mass

#=================================================================================================================#

# driver code (PUT THIS IN MAIN)
if __name__ == "__main__":
    """
    Section contains conversions to suitable units.
    Change this depending on units read from sensors.
    """
    a = 2.0 * np.pi * (6378388.0 + altitude) / 360.0    # approx lat. and long. (converted to metres)

    # calculate rate of change for x and y
    change_x = np.cumsum(a * np.cos(latitude * np.pi / 180.0) * np.hstack((0.0, np.diff(longitude))))
    change_y = np.cumsum(a * np.hstack((0.0, np.diff(latitude))))

    # sensor measurements
    sensor_measurements = np.vstack((change_x,
                                     change_y,
                                     heading*np.pi/180.0
                                     speed_x/3.6,
                                     speed_y/3.6,
                                     yawrate/180.0*np.pi,
                                     accel_x,
                                     accel_y,
                                     slip_ratio_FR,
                                     slip_ratio_RR,
                                     slip_ratio_FL,
                                     slip_ratio_RL))

    # initial state vector X_hat_1
    X_hat_t = np.matrix([[change_x[0],                   # initial x pos (metres)
                          change_y[0],                   # initial y pos (metres)
                          heading[0] * np.pi / 180.0,    # initial heading (radians)
                          speed_x[0] / 3.6,              # initial speed x component (m/s)
                          speed_y[0] / 3.6,              # initial speed y component (m/s)
                          yawrate[0] * np.pi / 180.0,    # initial yawrate (radians)
                          accel_x[0],                    # initial acceleration in x direction (m/s^-2) 
                          accel_y[0],                    # initial acceleration in y direction (m/s^-2)
                          slip_ratio_FR[0],              # slip ratio of front right wheel
                          slip_ratio_RR[0],              # slip ratio of rear right wheel
                          slip_ratio_FL[0],              # slip ratio of front left wheel
                          slip_ratio_RL[0]]])            # slip ratio of rear left

    X_hat_t = X_hat_t.T  # transpose to make 12x1

    # initial state covariance P_t
    P_t = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

    # CHANGE SLIP RATIO NOISE
    # process noise Q_t
    Q_t = np.diag([(1.0/50.0*0.5*8.8)**2,    # noise in gps (x component)
                   (1.0/50.0*0.5*8.8)**2,    # noise in gps (y component)
                   (1.0/50.0*0.1)**2,        # noise in heading
                   (1.0/50.0*8.8)**2,        # noise in speed (x component)
                   (1.0/50.0*8.8)**2,        # noise in speed (y component)
                   (1.0/50.0)**2,            # noise in yawrate
                    0.5**2,                  # noise in acceleration (x component)
                    0.5**2,                  # noise in acceleration (y component)
                    1.0,                     # noise in slip ratio (FR)
                    1.0,                     # noise in slip ratio (RR)
                    1.0,                     # noise in slip ratio (FL)
                    1.0])                    # noise in slip ratio (RL)

    # CHANGE SLIP RATIO NOISE
    # measurement noise R_t
    R_t = np.diag([5.0**2,    # noise in gps (x component)
                   5.0**2,    # noise in gps (y component)
                   0.1**2,    # noise in heading
                   3.0**2,    # noise in speed (x component)
                   3.0**2,    # noise in speed (y component)
                   0.1**2,    # noise in yawrate
                   1.0**2,    # noise in acceleration (x component)
                   1.0**2,    # noise in acceleration (y component)
                   1.0,       # noise in slip ratio (FR)
                   1.0,       # noise in slip ratio (RR)
                   1.0,       # noise in slip ratio (FL)
                   1.0])      # noise in slip ratio (RL)

    # x_pos, y_pos, head = [], [], []

    print("[+] UGRDV-21 - Velocity Estimation [+]\n")

    print("Workflow:")
    print("Running Extended Kalman Filter...")

    # run the algorithm
    extended_kalman_filter(X_hat_t, P_t, Q_t, R_t)

    print("\n[+] Execution terminated [+]")
