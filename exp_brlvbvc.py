from __future__ import print_function

from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite


class run_brlvbvc(ExperimentSuite):

    @property
    def train_weathers(self):
        return [1]

    @property
    def test_weathers(self):
        return [5]

    @property
    def collision_as_failure(self):
        return True

    @property
    def traffic_light_as_failure(self):
        return False

    def calculate_time_out(self, path_distance):
        """
        Function to return the timeout, in milliseconds,
        that is calculated based on distance to goal.
        This timeout is increased since stop for traffic lights is expected.
        """
        return ((path_distance / 1000.) / 5.) * 3600. + 20. + 100.

    def _poses_town01(self):
        """
        Each matrix is a new task.
        Use view_start_positions.py in Python Client to see positions
        """

        def _simple_poses_forward():
            return [[0, 13], [84, 40], [120, 106], [89, 133]]

        def _simple_poses_turn_left():
            return [[74, 66], [48, 37], [87, 100], [104, 82]]

        def _simple_poses_turn_right():
            return [[67, 75], [83, 103], [99, 86], [38, 49]]

        return [_simple_poses_forward(),
                _simple_poses_turn_left(),
                _simple_poses_turn_left(),
                _simple_poses_forward(),
                _simple_poses_turn_right(),
                _simple_poses_turn_right(),
                _simple_poses_forward(),
                _simple_poses_turn_left(),
                _simple_poses_turn_right(),
                _simple_poses_forward(),
                _simple_poses_turn_right(),
                _simple_poses_turn_left(),
                _simple_poses_forward(),
                _simple_poses_turn_left(),
                _simple_poses_turn_right(),
                _simple_poses_turn_left(),
                _simple_poses_forward(),
                _simple_poses_turn_left(),
                ]

    def _poses_town02(self):
        """
        Each matrix is a new task.
        Use view_start_positions.py in Python Client to see positions
        """
        def _simple_straight_forward():
            return [[64, 55], [54, 63], [51, 46], [45, 49], [29, 50], [61, 40]]

        def _simple_left_turn():
            return [[73, 62], [62, 7], [29, 43], [45, 79]]

        def _simple_right_turn():
            return [[78, 46], [44, 40], [18, 60], [60, 71]]

        return [_simple_straight_forward(),
                _simple_right_turn(),
                _simple_left_turn(),
                ]

    def build_experiments(self):
        """
        Creates the whole set f experiment objects,
        The experiments created depend on the selected Town.
        """

        #  width=500, height=350, pos_x= 2.0, pos_y=0.0, pos_z= 1.4, angle=-30.0
        cameraRGB = Camera('Camera', PostProcessing='SceneFinal')
        cameraRGB.set_image_size(500, 350)
        cameraRGB.set_position(2.0, 0.0, 1.4)
        cameraRGB.set_rotation(-30.0, 0.0, 0.)
        cameraRGB.set(FOV=100)

        camera = Camera('CameraSem', PostProcessing='SemanticSegmentation')
        camera.set_image_size(320, 180)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-30.0, 0.0, 0.)
        camera.set(FOV=100)

        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = []
            pedestrians_tasks = []
            for i in range(len(poses_tasks)):
                vehicles_tasks.append(0)
                pedestrians_tasks.append(0)

        experiment_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather,
                    QualityLevel=1
                    )

                conditions.set(SynchronousMode=True)
                conditions.set(DisableTwoWheeledVehicles=True)

                conditions.add_sensor(camera)
                conditions.add_sensor(cameraRGB)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                    )

                experiment_vector.append(experiment)

        return experiment_vector
