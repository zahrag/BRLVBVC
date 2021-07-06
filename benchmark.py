#!/usr/bin/env python3

"""
    Authors: Zahra Gharaee & Karl Holmquist
    This repository contains the codes of the vision-based vehicular control project applying
    a Bayesian approach to reinforcement learning designed and implemented in a virtually simulated environment
    available through the CARLA simulator (https://carla.org/). The paper is presented and published in the
    "Proceedings of the 2020 25th International Conference on Pattern Recognition (ICPR)" hold on 10-15 January 2021,
    virtually in Milan, Italy.
    IEEE link: https://ieeexplore.ieee.org/abstract/document/9412200
    Arxiv link: https://arxiv.org/abs/2104.03807
    Youtube link: https://www.youtube.com/watch?v=Y4SRHktFkug
    """

import argparse
import logging

from carla.driving_benchmark.driving_benchmark import run_driving_benchmark
from carla.sensor import Camera

from exp_brlvbvc import run_brlvbvc
from brlvbvc_agent import ZGH_BRL_Agent
from test_brlvbvc import testing_brlvbvc


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '--carla100',
        action='store_true',
        help='If you want to use the carla100 benchmark instead of the Basic one'
    )
    argparser.add_argument(
        '--run_brlvbvc',
        action='store_true',
        help='If you want to use the run_brlvbvc benchmark instead of the Basic one'
    )
    argparser.add_argument(
        '--test_brlvbvc',
        action='store_true',
        help='If you want to benchmark the test_exp instead of the Basic one'
    )
    argparser.add_argument(
        '--segment',
        action='store_true',
        help='If you want to run pspnet for semantic segmentation'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )

    args = argparser.parse_args()
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    if args.run_brlvbvc:
        experiment_suite = run_brlvbvc(args.city_name)
    elif args.test_brlvbvc:
        experiment_suite = testing_brlvbvc(args.city_name)
    elif args.carla100:
        experiment_suite = CARLA100(args.city_name)
    else:
        print(' WARNING: running the basic driving benchmark, to run for test_brlvbvc'
              ' experiment suites, you should run'
              ' python driving_benchmark_example.py --test_brlvbvc')
        experiment_suite = BasicExperimentSuite(args.city_name)

    agent = ZGH_BRL_Agent(segment_image=args.segment) 
    
    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
