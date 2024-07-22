from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
import os

def increment_indices(current_indices, max_indices):
    '''
    This function increments the indices-list specifying a particular hyperparameter combination used for an experiment.
    Each hyperparameter that can be adjusted is represented by an element in the list. It can be imagined as a number lock
    with different possible values for each position. The function increases the value for the last position by one,
    adjusting the positions in front of it in case of an overflow. 
    This allows the program to iterate through all possible hyperparameter combinations.
    '''
    current_indices[-1] += 1
    for i in reversed(range(len(current_indices))):
        if current_indices[i] > max_indices[i]:
            current_indices[i] = 0
            if i != 0:
                current_indices[i-1] +=1
        else:
            break
    return current_indices

def fetch_algorithm_parameters(algorithm_config, parameter_indices):
    '''Retrieves the specific list of parameters from the current parameter indices'''
    parameters = {k:v[parameter_indices[i]] for i, (k, v) in enumerate(algorithm_config.items())}
    return parameters

def create_dataframe_from_results(algorithm_config, parameter_values_list, metric_results_list, 
                                  solutions_list, num_of_experiments, METRIC_FEATURE_NAMES, algorithm_name):
    '''Stores the results obtained by an optimisation algorithm into a pandas dataframe after doing some pre-processing.'''
    #extract function names from parameter list
    for i in range(len(parameter_values_list)):
        parameter_values_list[i] = [elem.__name__ if callable(elem) else elem for elem in parameter_values_list[i]]

    #calculate the number of rows in metric_results_list that correspond to the same experiment
    num_rows_per_experiment = metric_results_list.shape[0] // num_of_experiments

    #extend parameter_values_list to have the same number of elements as metric_results_list
    parameter_values_list = list(itertools.chain.from_iterable(itertools.repeat(x, num_rows_per_experiment) for x in parameter_values_list))

    #extract column header names from algorithm config dict
    param_column_headers = list(algorithm_config.keys())
    param_column_headers = [str.split(key,"_and_") for key in param_column_headers] #split compound keys
    param_column_headers = list(itertools.chain.from_iterable(param_column_headers))#flatten list

    #convert numpy solution arrays to strings
    solutions_list = [np.array2string(x.flatten(), max_line_width=np.inf) for x in solutions_list]

    #create dataframes for metric values, parameter values, solutions and for the metadata valid for all conducted experiments
    df_metric_values = pd.DataFrame(metric_results_list, columns= METRIC_FEATURE_NAMES)
    df_param_values = pd.DataFrame(parameter_values_list, columns = param_column_headers)

    #concatenate columns of dataframes
    df_results = pd.concat([df_param_values,df_metric_values], axis=1)
    df_results = df_results.loc[(df_metric_values!=-1).any(axis=1)] #remove unused rows
    df_results = df_results.assign(solutions=solutions_list) #append solutions

    df_results["algorithm"] = algorithm_name #append information on which search algorithm was used
    return df_results

def add_metadata(df: pd.DataFrame, parameters, env_config_id):
    df["env_config_id"] = env_config_id

    for parameter_name, value in parameters.items():
        #tries setting the value, if that doesn't work, changes the datatype of the column to "object" and tries again
        try:
            df[parameter_name] = value
        except:
            df[parameter_name] = pd.Series([value] * len(df))
    return df

def gridsearch(algorithm, env, run_config: dict, seed: int = 11, csv_file_path: str = "data/", experiment_name: str = "experiment"):
    '''This function conducts gridsearch on a specific algorithm and environment in an effort to find optimal hyperparameters.
       The parameters to explore are defined in the run_config dictionary.
       The function generates a csv file of the evaluation results for each combination of hyperparameters
    '''

    #create directory to store experiment results if it is not there already
    if not os.path.isdir(csv_file_path):
        print("Directory",csv_file_path, "doesn't exist. Creating it now...")
        os.makedirs(csv_file_path)
    file_path = csv_file_path + experiment_name

    max_parameter_indices = np.array([len(x)-1 for x in run_config["init"].values()], dtype=int) #compute the maximum indices for each config parameter
    num_of_experiments = np.cumprod(max_parameter_indices + 1)[-1] #compute the number of experiments to run based on number of values in config
    
    #run all experiments
    df_list = []
    for env_config_id in tqdm(range(len(run_config["env"])), desc="Environment", position=1, leave=False):
        current_parameter_indices = np.zeros(len(run_config["init"]), dtype= int) #initialise current parameter indices to 0
        current_env_config = run_config["env"][env_config_id]# Old code: {k:v[env_config_id] for k,v in run_config["env"].items()}
        env.unwrapped.configure(current_env_config)
        for experiment_id in tqdm(range(num_of_experiments), desc="Experiments", position=2, leave=False):
            parameters = fetch_algorithm_parameters(run_config["init"], current_parameter_indices)
            obs, _ = env.reset()
            agent = algorithm(env = env, num_objectives = 2, seed = seed, observation_space_shape = obs[0].shape, num_actions = 5, objective_names=["speed_reward", "energy_reward"], **parameters)
            agent.train(**run_config["train"])
            agent.store_network(csv_file_path, f"{experiment_name}_config_{env_config_id}_exp{experiment_id}.pth")
            returns = agent.evaluate(**run_config["eval"])
            df = returns[0]
            df = add_metadata(df, parameters, env_config_id)
            df.to_csv(f"{file_path}_config_{env_config_id}_exp{experiment_id}.csv")
            df_list.append(df)
            #increment indices of the current algorithm's parameters
            current_parameter_indices = increment_indices(current_parameter_indices, max_parameter_indices)
    
    df = pd.concat(df_list)
    df.to_csv(f"{file_path}_merged.csv")