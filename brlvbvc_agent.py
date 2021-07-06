try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla.client import VehicleControl
from carla.agent import Agent

import matplotlib.pyplot as plt

import psutil
import gc
gc.enable()

import pickle
import time
import os

import ZGH_BRL_VBVC as ZGH


class ZGH_BRL_Agent(Agent):
    """
    Agent implementation class for using with Carla 8.4
    """
    def __init__(self, trained_model_pth=None, verbose=False,
                 start_as_test=False, save_models=True,
                 segment_image=False):
        super(ZGH_BRL_Agent, self).__init__()

        self.segment_image = segment_image
        self.save_models_at_end_of_phase = save_models
        self.verbose = verbose
        self.show_img = False
        self.trained_model_pth = trained_model_pth
        self.dirs = ['Reach goal', 'Unknown', 'Lane follow', 'Left', 'Right', 'Forward']
        self._dist_for_success = 2.0  # Also used by driving_benchmark.py for detecting success
        self.reset_counter = 0

        if self.trained_model_pth is not None:
            print("Loading model from: {}".format(self.trained_model_pth))
            self.load_model(self.trained_model_pth)
            print(self.info.episode)
            if start_as_test:
                print("The loaded model will be put in test mode (No learning)")
                for agent in self.learner:
                    agent.isLearning = False
                self.info.phase = 4
                self.info.episode = self.info.epi_max_III
                self.save_models_at_end_of_phase = False
            else:
                print("Model training will be resumed from checkpoint")
                # No need to do anything since everything is already loaded
        else:
            self.info, self.learner = ZGH.Initialization.init_all()

        self.info.was_reset = True
        self.info.reset_collision()

        self.reset = True

        self.frame_cnt = 0
        self.train_frq = 7
        self.intermediate_frq = 10
        self.last_control = VehicleControl()

        # Default value for master action (This will change during phase 1)
        self.master_action = 0

        self.old_target_dist = 0.

        if self.segment_image:
            import torch
            import torchvision.transforms as transforms
            import torch.nn as nn

            import sys
            sys.path.append("fcn")
            from fcn.prepare_psp import get_psp_resnet50_ade
            from fcn.prepare_EncNet import get_encnet_resnet101_ade

            self.use_psp = False

            pretrained = False
            freeze_up_to = None
            aux = True
            fusion_method = None
            norm_layer = nn.BatchNorm2d

            if self.use_psp:
                my_nclass = 9
                test_model_file = 'path_to_weights' + '/pspnet_001.pth'
                single_model, _ = get_psp_resnet50_ade(pretrained, my_nclass, freeze_up_to,
                                                       aux, norm_layer, fusion_method)
            else:
                my_nclass = 10
                test_model_file = 'path_to_weights' + '/EncNet_no_dataAug_epoch145.pth'
                single_model = get_encnet_resnet101_ade(my_nclass, pretrained, se_loss=True)

            single_model.cuda()
            bestparams = torch.load(test_model_file)
            single_model.load_state_dict(bestparams['model_state'])
            single_model.aux = False
            single_model.eval()
            del single_model.auxlayer

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Depth and Normal are using the same normalization values as for RBGs now.
            ])

            self.sem_model = single_model

        self.memory_used = psutil.virtual_memory().used

        if self.verbose:
            print("Initialization of BRL agent done!")

    def load_model(self, fname):
        data = pickle.load(open(fname, 'rb'))
        self.info = data['info']
        self.learner = data['agents']

    def save_model(self):
        fname = "{}/models/ZGH_phase{}_episodes{}_{}.pck".format(os.path.dirname(os.path.realpath(__file__)),
                                                                 self.info.phase, self.info.episode, time.time())
        data = {'info': self.info, 'agents': self.learner}

        pickle.dump(data, open(fname, 'wb'))

    def print_memory_used(self):
        print(psutil.virtual_memory().used - self.memory_used)

    def run_step(self, measurements, sensor_data, directions, target):
        """
        Defines the steps taken by the agent
        :param measurements: World measurements
        :param sensor_data: Data from specified sensors
        :param directions: Direction to take in intersection (Left, Right, Forward)
        :param target: ???
        :return: VehicleControl message
        """

        if True:

            self.frame_cnt += 1

            if measurements.player_measurements.intersection_offroad > 0.99:
                # Do nothing when outside of road
                return VehicleControl()

            # After each reset the timestamp is reset, wait until simulator is ready (t_delay ms)
            t_delay = 3000
            if measurements.game_timestamp < t_delay:
                # Less than X ms since episode start
                self.reset = True
                self.last_control = VehicleControl()
                self.info.was_reset = True
                return self.last_control
            elif self.reset:
                self.info.was_reset = True
                self.reset = False
                self.reset_counter += 1
                self.frame_cnt = 0

                self.old_target_dist = self._dist_to_target(measurements, target)

            if self.frame_cnt < self.train_frq:
                # Resend signal
                return self.last_control
            else:
                self.frame_cnt = 0

            # Get sensor data of interest

            if self.info.phase == 1 and self.frame_cnt % (self.train_frq * 20) == 0:
                self.master_action = np.random.choice([0, 3])

            if self.frame_cnt % self.train_frq == 0:

                if self.segment_image:
                    self.sem_image_GT = sensor_data.get('CameraSem', None)
                    if self.sem_image_GT is not None:
                        self.sem_image_GT = self.sem_image_GT.data

                    rgb_img = sensor_data.get('Camera')
                    if rgb_img is not None:
                        rgb_img = rgb_img.data.copy()  # np.expand_dims(rgb_img.data, axis=0)
                        import torch
                        with torch.no_grad():
                            output = self.sem_model(torch.unsqueeze(self.transform(rgb_img).cuda(), 0))

                            score_maps = output[0].data
                            _, predict = torch.max(score_maps, 1)
                            self.sem_image = predict.cpu().numpy().astype('uint8')

                        if self.use_psp:
                            # Process image to keep same labels (9 classes)
                            self.sem_image[self.sem_image == 7] = 10
                            self.sem_image[self.sem_image == 4] = 7
                            self.sem_image[self.sem_image == 1] = 4
                            self.sem_image[self.sem_image == 0] = 1
                            self.sem_image[self.sem_image == 6] = 9
                            self.sem_image[self.sem_image == 3] = 6
                            self.sem_image[self.sem_image == 8] = 11
                            self.sem_image[self.sem_image == 5] = 8
                            self.sem_image[self.sem_image == 255] = 3
                        else:
                            # Process image to keep same labels (10 classes)
                            self.sem_image[(self.sem_image >= 2) & (self.sem_image <= 9)] += 2
                            self.sem_image[self.sem_image == 255] = 3
                            self.sem_image[self.sem_image == 1] = 2
                            self.sem_image[self.sem_image == 0] = 1

                        # Clear some memory
                        del rgb_img
                        del output
                        del score_maps
                        del predict

                        # Specify ROI
                        dh = 40
                        dw = 5

                        # Shape: height x width
                        self.sem_image = np.transpose(self.sem_image, axes=(1, 2, 0))[:, :, 0]
                        self.sem_image = self.sem_image[dh:self.sem_image.shape[0]-dh, dw:self.sem_image.shape[1]-dw]

                    else:
                        self.sem_image = None
                else:
                    self.sem_image = sensor_data.get('CameraSem', None)
                    if self.sem_image is not None:
                        self.sem_image = self.sem_image.data

                if self.sem_image is not None:
                    if self.show_img:
                        plt.imshow(self.sem_image)
                        plt.waitforbuttonpress()

                # Depth Maps
                self.depth_image = sensor_data.get('CameraDepth', None)
                if self.depth_image is not None:
                    self.depth_image = self.depth_image.data
                brl_epi = self._progress(self.sem_image, self.depth_image, measurements, target, directions)

        gc.collect()
        return self.last_control

    def _progress(self, image, depth_image, measurements, target, directions):
        """
        Function for performing all in main if (train /predict)
        Will update the self.last_control
        image: Input image
        measurements; Object with all vehicle information
        target: Information about the target position
        directions: Top-level planning instructions (Forward, Backward, Left, Right)
        """
        # self.info.episode += 1
        action = None

        if self.info.episode < self.info.epi_max_II:
            if self.info.phase != 2:
                self.info.phase = 2
                print("\nStarting phase 2")
                self.learner[0].isLearning = True

        else:
            if self.info.phase != 4:
                print("\nStarting test phase")
                print("reset_counter: ", self.reset_counter)
                self.info.phase = 4
                self.learner[0].isLearning = False

        print("", end='\r')
        print("Episode: {}\t speed: {} \t direction: {}".format(self.info.episode,
                                                                measurements.player_measurements.forward_speed * 3.6,
                                                                self.dirs[int(directions)]), end=" ", flush=True)

        # Update state of all agents (agents.nx is calculated)
        self.learner[0].update_state(image, depth_image, self.segment_image)

        # Handle if simulation has been reset or in the start when no motor action performed
        if self.info.episode == 1 or self.info.episode == self.info.epi_max_II or self.info.was_reset:
            self.info.was_reset = False
            self.last_control = self._get_control_signal(0)

        else:
            # Based on the updated state (agent.nx),
            # motor_rewarding, br_learning, updating_brl_components and log_statistics are done
            dist = self._dist_to_target(measurements, target)
            self.learner = ZGH.Rewarding.get_motor_reward(self.info, self.learner, measurements, image,
                                                          self.old_target_dist - dist)
            self.old_target_dist = dist
            self.learner[0].perform_brl_learning()
            self.learner[0].update_learning_parameters()
            self.learner[0].update_log_statistics()

        # Update the current state as well as
        self.learner[0].x = self.learner[6].nx
        self.learner[0].update_components()

        self.learner, action, action_type = ZGH.DecisionMaking.action_selection(self.learner,
                                                                                self.dirs[int(directions)])

        if action is not None:
            self.last_control = self._get_control_signal(action)
            self.info.episode += 1

        if self.verbose:
            print("Episode nr. {}".format(self.info.episode), end="\n")
            print("Action: {}\tAction type: {}".format(action, action_type))
            print("Control msg: {}".format(self.last_control), end="\r")

        return self.info.episode

    @staticmethod
    def _dist_to_target(measurements, target):
        """
        Calculates the bird distance from current position to goal
        :param measurments:
        :param target:
        :return: Distance from player to target
        """
        lp = measurements.player_measurements.transform.location
        lt = target.location
        return np.sqrt((lp.x - lt.x) ** 2 + (lp.y - lt.y) ** 2 + (lp.z - lt.z) ** 2)

    @staticmethod
    def _get_control_signal(action):
        """
        Convert an action index to a VehicleControl() msg
        :param action: Action index
        :return: VehicleControl msg
        """
        control = VehicleControl()
        control.reverse = False
        control.steer = 0.
        control.throttle = 0.
        control.brake = 0.

        if action == 0:  # Fast forward
            control.throttle = 0.5
        elif action == 1:  # right turn
            control.steer = 0.4
            control.throttle = 0.35
        elif action == 2:  # left turn
            control.steer = -0.4
            control.throttle = 0.35
        elif action == 3:  # reverse
             control.reverse = True
             control.throttle = 0.4

        return control
