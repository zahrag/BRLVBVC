import numpy as np


def get_motor_reward(info, all_BRLearner, measurements, seg_image, target_dist=0.):

    # ------------------------- CHECK if Different Rewards Cases Occurred --------------------------
    speed = measurements.player_measurements.forward_speed * 3.6  # [km/h] - speed
    other_lane = measurements.player_measurements.intersection_otherlane  # [0,1] - percentage on other lane
    offroad = measurements.player_measurements.intersection_offroad  # [0,1] - percentage offroad
    collision_pedestrians = measurements.player_measurements.collision_pedestrians
    collision_vehicles = measurements.player_measurements.collision_vehicles
    collision_other = measurements.player_measurements.collision_other

    # Check if collision
    if collision_pedestrians - info.acc_collision_ped > 0 or \
            collision_vehicles - info.acc_collision_veh > 0 or \
            collision_other - info.acc_collision_oth > 0:
        collision_flag = True
        info.acc_collision_ped = collision_pedestrians
        info.acc_collision_veh = collision_vehicles
        info.acc_collision_oth = collision_other
        if info.verbose:
            print('\nAccident!\n')
    else:
        collision_flag = False

    # Check if offroad
    if offroad > 0.:
        offroad_flag = True
        if info.verbose:
            print('\nDriving off the road boundary!\n')
    else:
        offroad_flag = False

    # Check if wrong side of the road
    if other_lane > 0.:
        other_lane_flag = True
        if info.verbose:
            print('\nIntersecting the road lanes!\n')
    else:
        other_lane_flag = False

    # Check speed reward
    speed_max = 6
    speed_reward = 0.
    if speed < 0.:  # Driving Backward
        speed_reward = -5. * ((speed - speed_max) / speed_max) ** 2

    elif speed >= 0.:  # Driving Forward
        speed_reward = -10. * ((speed - speed_max) / speed_max) ** 2

    if speed == speed_max:
        if info.verbose:
            print('\nDriving Good!\n')

    # ---------------------------- Update the reward of all agents --------------------------
    for a in all_BRLearner:
        a.Reward = 0.  # Reset reward from last episode

        # Calculate amount of road in view
        road = np.sum(a.nx[0:-a.CN:a.fsz])
        if a.type == 'FOA':
            road = a.nx[0, 0] + a.nx[5, 0] + a.nx[10, 0] + a.nx[15, 0] + a.nx[20, 0] + a.nx[25, 0] +\
                   a.nx[1, 0] + a.nx[6, 0] + a.nx[11, 0] + a.nx[16, 0] + a.nx[21, 0] + a.nx[26, 0]
            # print('\nroad\n', road)

        if collision_flag is True:
            state_features_t = np.sum(a.nx[:a.num_grids*a.fsz].reshape(a.num_grids, a.fsz), axis=0)
            # Sum over the four patches of the image
            a.Reward = - 50.*(state_features_t[3]+state_features_t[4])

        elif np.any(a.nx[np.size(a.nx, 0)-3:np.size(a.nx, 0), 0] < [2, 1, 2]):
            dists = a.nx[np.size(a.nx, 0)-3:np.size(a.nx, 0), 0] / [2, 1, 2]
            min_dist = np.argmin(dists)
            a.Reward = -50 * (1 - dists[min_dist])

        elif offroad_flag:
            a.Reward = - 40. * offroad

        elif other_lane_flag:
            a.Reward = - 30. * other_lane

        else:
            # Control if reward should be based on speed or on delta distance to goal
            if True:
                a.Reward = speed_reward
            else:
                a.Reward = np.min([10. * target_dist, 10.]) - 10.

        if road < 0.5:
            a.Reward += 20*(road - 1)

    return all_BRLearner

