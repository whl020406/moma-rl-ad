from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
import os
from src.MOMA_DQN import MOMA_DQN


def increment_indices(current_indices, max_indices):
    current_indices[-1] += 1
    for i in reversed(range(len(current_indices))):
        if current_indices[i] > max_indices[i]:
            current_indices[i] = 0
            if i != 0:
                current_indices[i - 1] += 1
        else:
            break
    return current_indices


def fetch_algorithm_parameters(algorithm_config, parameter_indices):
    parameters = {k: v[parameter_indices[i]] for i, (k, v) in enumerate(algorithm_config.items())}
    return parameters


def create_dataframe_from_results(algorithm_config, parameter_values_list, metric_results_list,
                                  solutions_list, num_of_experiments, METRIC_FEATURE_NAMES, algorithm_name):
    for i in range(len(parameter_values_list)):
        parameter_values_list[i] = [elem.__name__ if callable(elem) else elem for elem in parameter_values_list[i]]

    num_rows_per_experiment = metric_results_list.shape[0] // num_of_experiments
    parameter_values_list = list(
        itertools.chain.from_iterable(itertools.repeat(x, num_rows_per_experiment) for x in parameter_values_list))

    param_column_headers = list(algorithm_config.keys())
    param_column_headers = [str.split(key, "_and_") for key in param_column_headers]
    param_column_headers = list(itertools.chain.from_iterable(param_column_headers))

    solutions_list = [np.array2string(x.flatten(), max_line_width=np.inf) for x in solutions_list]

    df_metric_values = pd.DataFrame(metric_results_list, columns=METRIC_FEATURE_NAMES)
    df_param_values = pd.DataFrame(parameter_values_list, columns=param_column_headers)

    df_results = pd.concat([df_param_values, df_metric_values], axis=1)
    df_results = df_results.loc[(df_metric_values != -1).any(axis=1)]
    df_results = df_results.assign(solutions=solutions_list)
    df_results["algorithm"] = algorithm_name
    return df_results


def add_metadata(df: pd.DataFrame, parameters, env_config_id, experiment_id):
    df["env_config_id"] = env_config_id
    df["experiment_id"] = experiment_id
    for parameter_name, value in parameters.items():
        try:
            df[parameter_name] = value
        except:
            df[parameter_name] = pd.Series([value] * len(df))
    return df


def gridsearch(algorithm, env, run_config: dict, seed: int = 11,
               csv_file_path: str = "data/",
               experiment_name: str = "experiment",
               only_evaluate: bool = False):

    if not os.path.isdir(csv_file_path):
        print("Directory", csv_file_path, "doesn't exist. Creating it now...")
        os.makedirs(csv_file_path)
    file_path = csv_file_path + experiment_name

    max_parameter_indices = np.array([len(x) - 1 for x in run_config["init"].values()], dtype=int)
    num_of_experiments = np.cumprod(max_parameter_indices + 1)[-1]

    num_reps = run_config["eval"].get("num_repetitions", 5)
    num_pts = run_config["eval"].get("num_points", 20)
    record_interval = int(max(1, num_pts / 10) * num_reps)

    summary_list = []
    detail_list = []
    loss_list = []

    for env_config_id in tqdm(range(len(run_config["env"])), desc="Environment", position=1, leave=False):
        current_parameter_indices = np.zeros(len(run_config["init"]), dtype=int)
        current_env_config = run_config["env"][env_config_id]
        env.unwrapped.configure(current_env_config)

        for experiment_id in tqdm(range(num_of_experiments), desc="Experiments", position=2, leave=False):
            parameters = fetch_algorithm_parameters(run_config["init"], current_parameter_indices)
            obs, _ = env.reset()

            try:
                # ✅ 创建 agent
                agent = algorithm(
                    env=env,
                    num_objectives=3,
                    seed=seed,
                    num_actions=5,
                    objective_names=["speed", "energy", "safety"],
                    **parameters
                )

                # ✅ 训练：传 train 配置+动态参数（gamma、batch_size）
                train_cfg = run_config["train"].copy()
                # 如果参数里包含 gamma / batch_size 也合并进去
                train_cfg.update({k: v for k, v in parameters.items()
                                  if k in ["gamma", "batch_size", "lr"]})

                # ✅ gridsearch 只跑训练（MOMA_DQN 没有 evaluate/store_network）
                if only_evaluate:
                    print("⚠ only_evaluate=True 但 MOMA_DQN 不支持 evaluate，跳过")
                else:
                    agent.train(**train_cfg)

                # ✅ 暂不执行 agent.evaluate / agent.store_network，避免 AttributeError
                # 未来如果需要，可以补上 evaluate & 保存接口

                # ✅ 这里只是防止 gridsearch 后面 concat 报错，填充空 df
                dummy_summary = pd.DataFrame([{"exp": experiment_id, "reward": 0}])
                dummy_detail = pd.DataFrame([{"exp": experiment_id, "step": 0, "reward": 0}])
                summary_list.append(dummy_summary)
                detail_list.append(dummy_detail)

            except Exception as e:
                print("The following error occurred during the experimentation. The current experiment configuration will be skipped")
                print(repr(e))

            current_parameter_indices = increment_indices(current_parameter_indices, max_parameter_indices)

    # ✅ 即使是 dummy 也得 concat，否则 ValueError: No objects to concatenate
    if summary_list:
        summary = pd.concat(summary_list)
        detail = pd.concat(detail_list)
    else:
        summary = pd.DataFrame()
        detail = pd.DataFrame()

    # ✅ 存个空 summary，防止后续报错
    summary.to_csv(f"{file_path}_merged_summary.csv")
    detail.to_csv(f"{file_path}_merged_detail.csv")
    print("✅ gridsearch completed (dummy summary saved)")
