# BRLVBVC
###### <h4> Authors: Zahra Gharaee (zahra.gharaee@liu.se) & Karl Holmquist (karl.holmquist@liu.se)

###### <h3> Overview 
This repository contains codes of a novel Bayesian approach to Reinforcement Learning of Vision Based Vehicular Control **BRLVBVC**, which addresses driving in a virtually simulated environment available through the [CARLA Simulator](https://carla.org/). Anyone interested in using **BRLVBVC** architecture and/or any of its components, please cite the following [article](https://ieeexplore.ieee.org/abstract/document/9412200)/[arxiv link](https://arxiv.org/abs/2104.03807), presented in ICPR virtual venue, which is available through the [link](https://www.youtube.com/watch?v=Y4SRHktFkug):
  
      @inproceedings {gharaee2020icpr} {
          author = {Zahra Gharaee and Karl Holmquist and Linbo He and Michael Felsberg},
            title = {A Bayesian Approach to Reinforcement Learning of Vision-Based Vehicular Control},
            booktitle = {2020 25th International Conference on Pattern Recognition (ICPR)},
            year = {2021},
            page = {3947--3954},
            Publisher = {IEEE},
            DOI = { 10.1109/ICPR48806.2021.9412200}
          }
        }
    

###### <h3> Prerequisities
Install python(3.5) + packages with conda from condaenv.yml.
Download binary for Carla (0.8.4) and install the API.
  
###### <h3> Run experiment
To allocate input data for training, validation and test experiments, varying conditions of the experimental setup including weather conditions, pedestrians, other vehicules, driving path and the town could be designed accordingly. Examples of such settings are available through python scripts exp_brlvbvc.p and test_brlvbvc.py. First start the Carla-server and based on your experiment set up, run the one of following command.
  
**Training**
```bash
python benchmark -c Town01 --run_brlvbvc
``` 
**Testing**
```bash
python benchmark -c Town02 --test_brlvbvc
```

**Result evaluation**
The collisions.py script contains codes for generating road histogram and estimate beta distributions for success rate and collision rate and the
eval.py script contains codes for estimating average distance between infractions.

Data used for input is generated in the corresponding _benchmark_results folder where the benchmark file was executed.

**Notes**
Weights for the trained EncNet model are not included because of the file size. Please find through link: https://github.com/zhanghang1989/PyTorch-Encoding for models and pre-trained weights.
 
###### <h3> Demo
A demo presenting the performance of the **BRLVBVC** architecture:  


https://user-images.githubusercontent.com/8222285/124627161-9bbd0100-de7f-11eb-9598-d3ef5c73917b.mp4



  
  
