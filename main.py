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
import matplotlib.pyplot as plt

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

def bytespdate2num(fmt, encoding='utf-8'):
    """ Formats date from dataset """
    def bytesconverter(b):
        s = b.decode(encoding)
        return (mdates.datestr2num(s))
    return bytesconverter


def plot():
    """ Plots the results from the EKF

        Outputs:
        - Result of the EKF algorithm
        - x-axis -> x position
        - y-axis -> y position
        - normal line -> EKF estimate
        - '+' -> GPS measurements
        - arrows -> estimated heading
    """
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


def calculate_long_tyre_stiffness(accel_x):
    """ Calcualtes longitudinal tyre stiffness

        Inputs:
        - accel_x (acceleration x-component)

        Outputs:
        - array of tyre longitudinal stiffnesses [FL, RL, FR, RR]
    """
    load_transfer_long = car_weight * accel_x * height_CoG / wheel_base    # update step (probably)

    front_left = front_right = rear_left = rear_right = 0.0    # loads on each tyre

    tyre_loads = [front_left, rear_left, front_right, rear_right]

    # tyre loads (longitudinal - x direction)
    tyre_loads = calculate_tyre_loads(tyre_loads, accel_x, load_transfer_long, car_weight)

    front_left_stiff = front_right_stiff = rear_left_stiff = rear_right_stiff = 0

    longitudinal_tyre_stiffnesses = [front_left_stiff, rear_left_stiff, front_right_stiff, rear_right_stiff]

    for i in range(len(longitudinal_tyre_stiffnesses)):
        longitudinal_tyre_stiffnesses[i] = fric_coefficient * tyre_loads[i] / 0.15
    
    return longitudinal_tyre_stiffnesses


def calculate_lat_tyre_stiffness(car_weight, height_CoG, track_width, accel_y, fric_coefficient):
    """ Calcualtes latitudinal tyre stiffness
        
        Inputs:
        - accel_y (acceleration y-component)

        Outputs:
        - array of tyre lateral stiffnesses [FL, RL, FR, RR]
    """
    load_transfer_lat = car_weight * accel_y * height_CoG / track_width

    lateral_tyre_stiffnesses = [front_left_stiff, rear_left_stiff, front_right_stiff, rear_right_stiff]

    front_left = front_right = rear_left = rear_right = 0.0    # loads one each tyre

    tyre_loads = [front_left, rear_right, front_right, rear_right]

    # tyre loads (latitudinal - y direction)
    tyre_loads = calculate_tyre_loads(tyre_loads, accel_y, load_transfer_lat, car_weight)

    front_left_stiff = front_right_stiff = rear_left_stiff = rear_right_stiff = 0

    longitudinal_tyre_stiffnesses = [front_left_stiff, rear_left_stiff, front_right_stiff, rear_right_stiff]

    # formula given in degrees but all measurements in radians?????
    for i in range(len(lateral_tyre_stiffnesses)):
        lateral_tyre_stiffnesses[i] = fric_coefficient * tyre_loads[i] / np.rad2deg(15*np.pi/180)
    
    return lateral_tyre_stiffnesses


def calculate_yaw_of_inertia(mass, wheel_base, track_width):
    """ Calculates yaw moment of inertia

        Inputs:
        - mass (mass of the car)
        - wheel base (horizontal distance between centres of front and rear wheels)
        - track_width (distance between rear wheels)

        Ouputs:
        - yaw_moment_of_inertia = mass x (wheel base^2 + track width^2) / 12.0
    """
    return mass * (wheel_base**2 + track_width**2) / 12.0


def f_M_formula(delta, vel_x, vel_y, longitudinal_tyre_stiffnesses, slip_ratio, lateral_tyre_stiffnesses, slip_angle):
    # F_m Forumla?? #

    slip_angle = delta - np.arctan(vel_x / vel_y)

    # denotes front wheels
    F_x1 = longitudinal_tyre_stiffnesses * slip_ratio
    F_x2 = F_x1    # not sure on this

    # denotes rear wheels
    F_y1 = lateral_tyre_stiffnesses * slip_angle
    F_y2 = F_y1    # not sure on this

    length_front = 1    # distance from front to centre
    length_rear = 1     # distance from rear to centre

    f_M = (1/yaw_of_inertia)*(length_front*(F_x1*np.sin(slip_angle)+F_y1*np.cos(slip_angle)-length_rear*F_y2))

    return f_M


def calculate_inertia_of_wheel(height_CoG, mass):
    """ Calculates moment of inertia of wheel

        Inputs:
        - height_CoG (height from the car's centre of gravity)
        - mass (mass of the car)

        Ouputs:
        - inertia_of_wheel = 0.5 x wheel_radius^2 x mass
    """
    wheel_radius = height_CoG

    return 0.5 * wheel_radius**2 * mass


def prediction(X_hat_t_1, P_t_1, Q_t):
    """ Estimates the next value for the state vector

        Inputs:
        - X_hat_t_1 (12x12 matrix: the previous measurement)
        - P_t_1 (12x6 matrix: the previous measurement's covariance matrix)
        - Q_t (12x12 matrix: the measurement noise matrix)
        
        Outputs:
        - X_hat_t (12x12 matrix:  the predicted value of the state estimate)
        - P_t_1 (12x6 matrix: the predicted state estimate's covariance matrix)
    """
    # I DON'T KNOW IF I SHOULD MAKE THESE GLOBAL OR NOT
    fric_coefficient = 1.5
    wheel_base = 1.53    # UGR's car wheel_base (m)
    height_CoG = 0.26    # UGR's car w/ driver (m) (300mm w/ driver 250-260mm w/o driver)
    mass = 201.5    # 201.5kg w/o driver and 227kg (2018 UGR)
    car_weight = mass * 9.8
    track_width = 1.201    # IMECHE's DV car (m) - distance between rear wheels

    length_front = 1    # distance from front to centre
    length_rear = 1     # distance from rear to centre


    yaw_of_inertia = calculate_yaw_of_inertia(mass, wheel_base, track_width)


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

    # Process Model stuff
    # P_ab1 = X_hat_t[3]*np.cos(X_hat_t[2]) - (X_hat[4]*np.sin(X_hat_t[2]))
    # P_ab2 = X_hat[4]*np.sin(X_hat_t[2]) + X_hat_t[3]*np.cos(X_hat_t[2])
    # P_ab3 = (1/m)*(force_rear_x-force_front_y*np.sin(dt)+m*X_hat_t[4]*X_hat_t[5])
    # P_ab4 = (1/m)*(force_rear_y+force_front_x*np.cos(dt)-m*X_hat_t[3]*X_hat_t[5])
    # P_ab5 = (1/inertia_x)*(force_front_y*load_front*np.cos(dt)-force_rear_y*load_rear+something)

    J_13 = (-X_hat_t[3]*np.sin(X_hat_t[2]))-(X_hat_t[4]*np.cos(X_hat_t[2]))
    J_14 = np.cos(X_hat_t[2])
    J_15 = -np.sin(X_hat_t[2])

    J_23 = (X_hat_t[3]*np.cos(X_hat_t[2]))-(X_hat_t[4]*np.sin(X_hat_t[2]))
    J_24 = np.sin(X_hat_t[2])
    J_25 = np.cos(X_hat_t[2])

    J_45 = X_hat_t[5]
    J_46 = X_hat_t[4]

    J_54 = -X_hat_t[5]
    J_56 = -X_hat_t[3]

    J_64 = (length_front*np.cos(dt)-length_rear)*((car_weight*accel_y*height_CoG*speed_y)/(yaw_of_inertia*track_width*(speed_x**2+speed_y**2)))
    J_65 = J_64

    if (X_hat_t[6] > 0):
        # car_weight +/- depending on +'ve/-'ve accel    
        J_67 = (length_front*fric_coefficient*car_weight*height_CoG/0.3*yaw_of_inertia*wheel_base)*X_hat_t[8]*np.sin(dt)
        J_68 = (length_front*np.cos(dt)-length_front)*((car_weight*height_CoG*(dt-np.arctan(speed_y/speed_x))/(yaw_of_inertia*track_width)))
        # car_weight +/- depending on +'ve/-'ve accel
        J_69 = (length_front/yaw_of_inertia)*(fric_coefficient*((car_weight)+((car_weight*X_hat_t[6]*height_CoG)/wheel_base)/0.3)*X_hat_t[8]*np.sin(dt))
    elif (X_hat_t[6] < 0):
        # car_weight +/- depending on +'ve/-'ve accel    
        J_67 = -(length_front*fric_coefficient*car_weight*height_CoG/0.3*yaw_of_inertia*wheel_base)*X_hat_t[8]*np.sin(dt)
        J_68 = (length_front*np.cos(dt)-length_front)*((car_weight*height_CoG*(dt-np.arctan(speed_y/speed_x))/(yaw_of_inertia*track_width)))
        # car_weight +/- depending on +'ve/-'ve accel
        J_69 = (length_front/yaw_of_inertia)*(fric_coefficient*((car_weight)-((car_weight*X_hat_t[6]*height_CoG)/wheel_base)/0.3)*X_hat_t[8]*np.sin(dt))

    J_97 = -(1/speed_FL)*(X_hat_t[8]+1)
    J_99 = ((lateral_tyre_stiffnesses[0]*radius)/inertia_of_wheel)-(X_hat_t[6]*(1/speed_FL))

    J_107 = -(1/speed_FR)*X_hat_t[9]
    J_1010 = ((lateral_tyre_stiffnesses[2]*radius)/inertia_of_wheel)-(X_hat_t[6]*(1/speed_FR))

    J_117 = -(1/speed_RL)*X_hat_t[10]
    J_1111 = ((lateral_tyre_stiffnesses[1]*radius)/inertia_of_wheel)-(X_hat_t[6]*(1/speed_RL))

    J_127 = -(1/speed_RR)*X_hat_t[11]
    J_1212 = ((lateral_tyre_stiffnesses[3]*radius)/inertia_of_wheel)-(X_hat_t[6]*(1/speed_RR))

    # jacobian matrix of A (CHANGE NUMBERS)
    J_A = np.matrix([[1.0, 0.0, J_13, J_14, J_15,  0.0,   0.0,   0.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 1.0, J_23, J_24, J_25,  0.0,   0.0,   0.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  1.0,  0.0,  0.0,  1.0,   0.0,   0.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0,  1.0, J_45, J_46,   1.0,   0.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0, J_54,  1.0, J_56,   0.0,   1.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0, J_64, J_65,  1.0,  J_67,  J_68, J_69,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0,   1.0,   0.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0,   0.0,   1.0,  0.0,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  J_97,   0.0, J_99,    0.0,    0.0,    0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, J_107,   0.0,  0.0, J_1010,    0.0,    0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, J_117,   0.0,  0.0,    0.0, J_1111,    0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, J_127,   0.0,  0.0,    0.0,    0.0, J_1212]])
    
    # error covariance in measurements (12x12 matrix)
    P_t_1 = J_A * P_t_1 * J_A.T + Q_t

    return X_hat_t, P_t_1


def update(X_hat_t, P_t, Z_t, R_t):
    """ Updates the prediction accounting for previous values

        Inputs:
        - X_hat_t (12x12 matrix: the current state estimate)
        - P_t (12x6 matrix: the current state estimate's covariance matrix)
        - Z_t (the conditioning matrix)
        - R_t (12x12 matrix: the measurement noise matrix)

        Outputs:
        - X_t (12x12 matrix: the new updated state estimate)
        - P_t (12x6 matrix: the new updated state estimate's covariance matrix)
    """

    # I DON'T KNOW IF I SHOULD MAKE THESE GLOBAL OR NOT
    fric_coefficient = 1.5
    wheel_base = 1.53    # UGR's car wheel_base (m)
    height_CoG = 0.26    # UGR's car w/ driver (m) (300mm w/ driver 250-260mm w/o driver)
    mass = 201.5    # 201.5kg w/o driver and 227kg (2018 UGR)
    car_weight = mass * 9.8
    track_width = 1.201    # IMECHE's DV car (m) - distance between rear wheels

    length_front = 1    # distance from front to centre
    length_rear = 1     # distance from rear to centre

    hx = np.matrix([[float(X_hat_t[0])], [float(X_hat_t[1])], [float(X_hat_t[3])],
                    [float(X_hat_t[4])], [float(X_hat_t[5])], [float(X_hat_t[6])],
                    [float(X_hat_t[7])], [float(X_hat_t[8])], [float(X_hat_t[9])],
                    [float(X_hat_t[10])], [float(X_hat_t[11])]])
 
    # change this (check maths obvs not correct)
    J_H = np.matrix([[1.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  1.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  1.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
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
        - X_hat_t (12x12 matrix: the initial state vector)
        - P_t (12x6 matrix: the initial state vector's covariance matrix)
        - Q_t (12x12 matrix: the process noise matrix)
        - R_t (12x12: the measurement noise matrix)

        Outputs:
        - X_t (12x12 matrix: the final state estimate)
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

# driver code (PUT THIS IN MAIN)
if __name__ == "__main__":
    """
    Section contains conversions to suitable units.
    Change this depending on units read from sensors.
    """

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
    date, time, millis, rollrate, pitchrate, yawrate, roll, pitch, yaw, speed, accel_x, accel_y, accel_z, heading, latitude, longitude, altitude, slip_ratio_FL, slip_ratio_FR, slip_ratio_RL, slip_ratio_RR = np.loadtxt(PATH, delimiter=',', unpack=True, 
                    converters={0: bytespdate2num('%Y-%m-%d')}, skiprows=1)
    
    a = 2.0 * np.pi * (6378388.0 + altitude) / 360.0    # approx lat. and long. (converted to metres)

    # calculate rate of change for x and y
    change_x = np.cumsum(a * np.cos(latitude * np.pi / 180.0) * np.hstack((0.0, np.diff(longitude))))
    change_y = np.cumsum(a * np.hstack((0.0, np.diff(latitude))))

    speed_x = speed
    speed_y = speed

    # sensor measurements
    sensor_measurements = np.vstack((change_x,
                                     change_y,
                                     heading*np.pi/180.0,
                                     speed_x/3.6,
                                     speed_y/3.6,
                                     yawrate/180.0*np.pi,
                                     accel_x,
                                     accel_y,
                                     slip_ratio_FL,
                                     slip_ratio_FR,
                                     slip_ratio_RL,
                                     slip_ratio_RR))

    # initial state vector X_hat_1
    X_hat_t = np.matrix([[change_x[0],                   # initial x pos (metres)
                          change_y[0],                   # initial y pos (metres)
                          speed_x[0] / 3.6,              # initial speed x compone
                          heading[0] * np.pi / 180.0,    # initial heading (radians)nt (m/s)
                          speed_y[0] / 3.6,              # initial speed y component (m/s)
                          yawrate[0] * np.pi / 180.0,    # initial yawrate (radians)
                          accel_x[0],                    # initial acceleration in x direction (m/s^-2) 
                          accel_y[0],                    # initial acceleration in y direction (m/s^-2)
                          slip_ratio_FL[0],              # slip ratio of front left wheel
                          slip_ratio_FR[0],              # slip ratio of front right wheel
                          slip_ratio_RL[0],              # slip ratio of rear left wheel
                          slip_ratio_RR[0]]])            # slip ratio of rear left wheel

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
                    1.0,                     # noise in slip ratio (FL)
                    1.0,                     # noise in slip ratio (FR)
                    1.0,                     # noise in slip ratio (RL)
                    1.0])                    # noise in slip ratio (RR)

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
                   1.0,       # noise in slip ratio (FL)
                   1.0,       # noise in slip ratio (FR)
                   1.0,       # noise in slip ratio (RL)
                   1.0])      # noise in slip ratio (RR)

    # x_pos, y_pos, head = [], [], []

    print("[+] UGRDV-21 - Velocity Estimation [+]\n")

    print("Workflow:")
    print("Running Extended Kalman Filter...")

    # run the algorithm
    extended_kalman_filter(X_hat_t, P_t, Q_t, R_t)

    print("\n[+] Execution terminated [+]")
