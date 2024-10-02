# MOMA-DQN
This repositority contains the implementation of a multi-agent multi-objective reinforcement learning algorithm based on Q-learning, intended to be used for the autonomous driving domain

![Baseline Algorithm](https://github.com/franzherm/moma-rl-ad/blob/main/videos/baseline_algo_ego_reward_CVR_0.1.gif)

The algorithm was tested on a highway traffic network. The requirements.txt contains the required python modules to be installed. The following section contains a description of the project file structure. The code was programmed using python 3.12.3


data/ -- contains the data obtained from running the experiments.

src/ -- contains the source code of the implemented algorithm, road network and auxiliary classes

videos/ -- contains selected videos of trained agents

root -- contains the jupyter notebook files used to analyse the experiments and scripts for running experiments.


To run the experiments for the highway scenario, execute the file **marl_highway_gridsearch.py**. The data is going to be stored in the data/ folder. Make sure that all the modules from the **requirements.txt** file have been installed!

To analyse experiments for this environment, run the cells of the jupyter notebook file **moma_highway_final_analysis.ipynb*
