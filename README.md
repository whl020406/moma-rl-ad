# MOMA-DQN
This repositority contains the implementation of a multi-agent multi-objective reinforcement learning algorithm based on Q-learning, intended to be used for the autonomous driving domain

The algorithm was tested on a highway traffic network. The requirements.txt contains the required python modules to be installed. The following section contains a description of the project file structure.


data/ -- contains the data obtained from running the experiments. It can be downloaded from the following link using university of exeter login credentials:

src/ -- contains the source code of the implemented algorithm, road network and auxiliary classes

root -- contains the jupyter notebook files used to analyse the experiments and scripts for running experiments.


To run the experiments for the highway scenario, execute the file **marl_highway_gridsearch.py**. The data is going to be stored in the data/ folder. Make sure that all the modules from the **requirements.txt** file have been installed!

To analyse experiments for this environment, run the cells of the jupyter notebook file **moma_highway_final_analysis_3rd_try.ipynb**

All the other notebook files in the root folder are used to execute and evaluate prelimiary experiments.