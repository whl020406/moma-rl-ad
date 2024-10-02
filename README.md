# MOMA-DQN
This repositority contains the implementation of a multi-agent multi-objective reinforcement learning algorithm based on Q-learning, intended to be used for the autonomous driving domain

The algorithm was tested on a highway traffic network. The requirements.txt contains the required python modules to be installed. The following section contains a description of the project file structure. The code was programmed using python 3.12.3


data/ -- contains the data obtained from running the experiments. It can be downloaded from the following link using university of exeter login credentials: 


[Link to experiment data](https://universityofexeteruk-my.sharepoint.com/:u:/g/personal/fh418_exeter_ac_uk/Eab1cLRSA01AlbDs6piN7_oBSnT5xXzDRdP1dcqP3-msRg?e=qX7j6Y)


The zip file contains a folder and a separate csv file. Extract these elements directly into the data folder instead of transforming the zip itself into a folder.
src/ -- contains the source code of the implemented algorithm, road network and auxiliary classes

root -- contains the jupyter notebook files used to analyse the experiments and scripts for running experiments.


To run the experiments for the highway scenario, execute the file **marl_highway_gridsearch.py**. The data is going to be stored in the data/ folder. Make sure that all the modules from the **requirements.txt** file have been installed!

To analyse experiments for this environment, run the cells of the jupyter notebook file **moma_highway_final_analysis_3rd_try.ipynb*
Make sure to download and extract the data of the experiment results beforehand using the link above if you don't want to run the experiments yourself!
