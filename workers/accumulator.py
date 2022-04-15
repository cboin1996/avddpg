from distutils.command import config
from logging import root
from operator import concat
from this import d
from typing import List, Optional
import pandas as pd
import numpy as np
from src import config, util
from src.env import env
import os, sys, shutil
import glob
import matplotlib.pyplot as plt

AGG_FNAME = "aggregation.csv"
AVERAGED_DF = "averaged.csv"

def generate_reward_plot(n_vehicles: int, timestamp: int, mode_limit: int) -> bool:
    """Accumulates and plots across multiple seeds, averaging the reward value
    across the seed and plotting with a shaded error bar.

    Args:
        mode (int): A value of 0 or 1 dictates which csv file to use for aggregation.
        0 = not averaged, 1 = averaged, 2 = fed weightings, 3 = fw percents
    """
    accum_path = os.path.join(sys.path[0], config.Config.report_dir, f"ACCUM_{timestamp}")
    os.mkdir(accum_path)
    # iterate modes 0,1..3
    for i in range (mode_limit+1):
        df = get_data(i, accum_path)
        df = transform(df, n_vehicles,i, accum_path)
        plot(df, n_vehicles, i, accum_path)
    return True


def get_data(mode: int, accum_path: str) -> pd.DataFrame:
    """Aggregates all csv's into a single csv file and writes to disk. Returns the object in a pd.DataFrame.
    """
    # TODO: Investigate why this function is not importing data correctly and duplicates the first entry!
    path_lst = glob.glob(os.path.join(sys.path[0], config.Config.res_dir, "*"))
    df = None
    for i, root_path in enumerate(path_lst):
        conf_path = os.path.join(root_path, config.Config.param_path)
        dir_name = os.path.basename(os.path.normpath(root_path))
        if not os.path.exists(os.path.join(accum_path, dir_name)):
            os.mkdir(os.path.join(accum_path, dir_name))
        conf = util.config_loader(conf_path)
        # heres where you can point to different files. 
        if mode == 0:
            data_path = conf.ep_reward_path % (conf.random_seed)
        elif mode == 1:
            data_path = conf.avg_ep_reward_path % (conf.random_seed)
        elif mode == 2 or mode == 3:
            if not hasattr(conf, "frl_weighted_avg_parameters_path"):
                raise ValueError(f"Mode {mode} is invalid as attribute frl_weighted_avg_parameters_path does not exist in config!")
            data_path = conf.frl_weighted_avg_parameters_path % (conf.random_seed)

        if df is None:
            df = pd.read_csv(os.path.join(root_path, data_path))

        shutil.copy(conf_path, os.path.join(accum_path, dir_name, config.Config.param_path))
        shutil.copy(os.path.join(root_path, data_path), os.path.join(accum_path, dir_name, 
                                                                    data_path))

        if df is not None and i > 0:
            df = df.append(pd.read_csv(os.path.join(root_path, data_path)))
        df_out = df.rename(columns={df.columns[0] : env.TRAINING_EPISODE_COLNAME})

    util.write_csv_from_df(df_out,os.path.join(accum_path, get_mode_tag(mode,AGG_FNAME)))
    return df_out

def transform(df, n_vehicles: int, mode: int, accum_path: str):
    """Performs a transformation on the given df, producing the averaged result

    Args:
        df (pd.DataFrame): the pandas dataframe
        n_vehicles (int): the number of assumed vehicle columns in the data frame.
    """
    # Get the number of platoons out of the data set.  Will return a single dataframe averaged for vehicles aross the seeds.
    df_c = df.copy()
    df_c.drop(columns=[env.SEED_COL], inplace=True)
    if mode == 1:
        df_c.drop(columns=[env.EPISODIC_REWARD_AVGWINDOW_COL], inplace=True)
    if mode == 2:
        df_c.drop(columns=get_fws_colnames(n_vehicles) + get_fwpct_colnames(n_vehicles), inplace=True)
    if mode == 3:
        df_c.drop(columns=get_fws_colnames(n_vehicles) + get_vehicle_colnames(n_vehicles), inplace=True)

    grouped_df = df_c.groupby([env.TRAINING_EPISODE_COLNAME, env.PLATOON_COL])
    avg_df = grouped_df.agg([np.mean, np.std])
    util.write_csv_from_df(avg_df,os.path.join(accum_path, get_mode_tag(mode,AVERAGED_DF)))
    return avg_df

def plot(df, n_vehicles: int, mode: int, accum_path: str):
    n_platoons = len(set(df.index.get_level_values(1)))

    for p_idx in range(n_platoons):
        fig, ax = plt.subplots()
        sub_df = df.xs(p_idx+1, level=env.PLATOON_COL) # Select out data pertinent to the platoon idx, p_id
        if mode == 0 or mode == 1:
            col_to_plot = "Vehicle"
        elif mode == 2:
            col_to_plot = "Vehicle Weighting"
        elif mode == 3:
            col_to_plot = "Vehicle Weighting in Percent"

        sub_df.columns.names = [col_to_plot, None]
        sub_df = sub_df.stack(col_to_plot).reset_index().sort_values(env.TRAINING_EPISODE_COLNAME).reset_index(drop=True)
        for i, v in sub_df.groupby(col_to_plot):
            ax.plot(v[env.TRAINING_EPISODE_COLNAME], v['mean'], linewidth=get_plot_weight(mode), label=v[col_to_plot].unique()[0])
            ax.fill_between(v[env.TRAINING_EPISODE_COLNAME],
                            v['mean'] - v['std'], 
                            v['mean'] + v['std'], alpha=0.35)
            ax.set_xlabel("Training episode")
            ax.set_ylabel(get_y_axis_title(mode))
            ax.legend()
        plt.rcParams.update({'font.size': 14})
        plt.tight_layout()
        plt.savefig(os.path.join(accum_path, get_mode_tag(mode,f"pl{p_idx+1}.svg")), dpi=150)

def get_vehicle_colnames(n_vehicles: int):
    return [env.VEHICLE_COL % (v_idx) for v_idx in range(1, n_vehicles+1)]

def get_fws_colnames(n_vehicles: int):
    return [env.FED_WEIGHT_SUM_COL % (v_idx) for v_idx in range(1, n_vehicles+1)]

def get_fwpct_colnames(n_vehicles: int):
    return [env.FED_WEIGHT_PCT_COL % (v_idx) for v_idx in range(1, n_vehicles+1)]


def get_mode_tag(mode, s) -> str:
    if mode == 0:
        return "ep_reward_" + s
    elif mode == 1:
        return "avgep_reward_" + s
    elif mode == 2:
        return "fw_" + s
    elif mode == 3:
        return "fws_" + s
    else:
        raise ValueError(f"No such mode exists {mode}")        
def get_y_axis_title(mode) -> str:
    if mode == 0:
        return "Average episodic reward"
    elif mode == 1:
        return "Cumulative average episodic reward"
    elif mode == 2:
        return "Federated Weighting"
    elif mode == 3:
        return "Federated Weighting Percent"
    else:
        raise ValueError(f"No such mode exists {mode}")
def get_plot_weight(mode) -> int:
    if mode == 0:
        return 0.35
    elif mode == 1:
        return 0.5
    elif mode == 2:
        return 0.5
    elif mode == 3:
        return 0.5
    else:
        raise ValueError(f"No such mode exists {mode}")
 