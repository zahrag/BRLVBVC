import json
import numpy as np


def get_metrics(filename, weathers):
    with open(filename, 'r') as f:
        datastore = json.loads(f.readlines()[0])
        total_km = {}
        intersection_offroad = {}
        intersection_otherlane = {}
        collision_other = {}
        collision_pedestrians = {}
        collision_vehicles = {}
        metrics = {}
        for setting in weathers:
            # Test or train
            total_km[setting] = np.zeros(4)
            intersection_offroad[setting] = np.zeros(4)
            intersection_otherlane[setting] = np.zeros(4)
            collision_other[setting] = np.zeros(4)
            collision_pedestrians[setting] = np.zeros(4)
            collision_vehicles[setting] = np.zeros(4)
            for weather in weathers[setting]:
                total_km[setting] += np.asarray(datastore["driven_kilometers"][weather])
                inter_off = [np.sum(task) for task in datastore["intersection_offroad"][weather]]
                inter_oth = [np.sum(task) for task in datastore["intersection_otherlane"][weather]]
                intersection_offroad[setting] += np.asarray(inter_off)
                intersection_otherlane[setting] += np.asarray(inter_oth)

                coll_ped = [np.sum(task) for task in datastore["collision_pedestrians"][weather]]
                coll_veh = [np.sum(task) for task in datastore["collision_vehicles"][weather]]
                coll_oth = [np.sum(task) for task in datastore["collision_other"][weather]]
                collision_other[setting] += np.asarray(coll_oth)
                collision_pedestrians[setting] += np.asarray(coll_ped)
                collision_vehicles[setting] += np.asarray(coll_veh)

            # Calculate mean distance between incidents:
            mean_distance_offroad = np.sum(total_km[setting]) / np.sum(intersection_offroad[setting])
            mean_distance_otherlane = np.sum(total_km[setting]) / np.sum(intersection_otherlane[setting])
            mean_distance_coll_other = np.sum(total_km[setting]) / np.sum(collision_other[setting])
            mean_distance_coll_vehicle = total_km[setting][-1] / collision_vehicles[setting][-1]
            mean_distance_coll_pedestrian = total_km[setting][-1] / collision_pedestrians[setting][-1]

            print("\tCondition: ", setting)
            print("Mean distance between otherlane: \t\t\t", mean_distance_otherlane)
            print("Mean distance between offroads: \t\t\t", mean_distance_offroad)
            print("Mean distance between collision w other:\t\t", mean_distance_coll_other)
            print("Mean distance between collision w vehicle:\t\t", mean_distance_coll_vehicle)
            print("Mean distance between collision w pedestrian:\t", mean_distance_coll_pedestrian)
            print("Distances: \t", total_km[setting], "\ttotal dist: \t", np.sum(total_km[setting]))


if __name__ == '__main__':
    weathers = {"train": ['1.0', '3.0', '6.0', '8.0'], "test": ['4.0', '14.0']}
    print("\n Weather conditions in train/test: ")
    print("Train weather: ", weathers["train"])
    print("Test weather: ", weathers["test"])

    print("Use get_metric() to evaluate the average distance between infractions")
