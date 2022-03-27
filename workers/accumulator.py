from distutils.command import config
from logging import root
from operator import concat
import pandas as pd
import numpy as np
from src import config, util
from src.env import env
import os, sys, shutil
import glob
import matplotlib.pyplot as plt

AGG_FNAME = "aggregation.csv"
AGG_AVG_FNAME = "cumavg_aggregation.csv"
AVERAGED_DF = "averaged.csv"
def generate_reward_plot(n_vehicles: int, timestamp: int) -> bool:
    """Accumulates and plots across multiple seeds, averaging the reward value
    across the seed and plotting with a shaded error bar.

    Args:
        mode (int): A value of 0 or 1 dictates which csv file to use for aggregation.
        0 = not averaged, 1 = averaged
    """
    accum_path = os.path.join(sys.path[0], config.Config.report_dir, f"ACCUM_{timestamp}")
    os.mkdir(accum_path)
    df = get_data(0, accum_path)
    df = transform(df, n_vehicles,0, accum_path)
    plot(df, n_vehicles, 0, accum_path)
    df = get_data(1, accum_path)
    df = transform(df, n_vehicles,1, accum_path)
    plot(df, n_vehicles, 1, accum_path)
    return True


def get_data(mode: int, accum_path: str) -> pd.DataFrame:
    """Aggregates all csv's into a single csv file and writes to disk. Returns the object in a pd.DataFrame.
    """
    path_lst = glob.glob(os.path.join(sys.path[0], config.Config.res_dir, "*"))
    df = None
    for root_path in path_lst:
        conf_path = os.path.join(root_path, config.Config.param_path)
        dir_name = os.path.basename(os.path.normpath(root_path))
        if not os.path.exists(os.path.join(accum_path, dir_name)):
            os.mkdir(os.path.join(accum_path, dir_name))
        conf = util.config_loader(conf_path)
        if mode == 0:
            data_path = conf.ep_reward_path % (conf.random_seed)
        else:
            data_path = conf.avg_ep_reward_path % (conf.random_seed)

        if df is None:
            df = pd.read_csv(os.path.join(root_path, data_path))

        shutil.copy(conf_path, os.path.join(accum_path, dir_name, config.Config.param_path))
        shutil.copy(os.path.join(root_path, data_path), os.path.join(accum_path, dir_name, 
                                                                    data_path))

        if df is not None:
            df = df.append(pd.read_csv(os.path.join(root_path, data_path)))
        df_out = df.rename(columns={df.columns[0] : env.EPISODIC_REWARD_COLNAME})

    agg_fname = get_agg_fname(mode)
    util.write_csv_from_df(df_out,os.path.join(accum_path, get_mode_tag(mode,agg_fname)))
    return df_out

def transform(df, n_vehicles: int, mode: int, accum_path: str):
    """Performs a transformation on the given df, producing the averaged result

    Args:
        df (pd.DataFrame): the pandas dataframe
        n_vehicles (int): the number of assumed vehicle columns in the data frame.
    """
    # Get the number of platoons out of the data set.  Will return a single dataframe averaged for vehicles aross the seeds.
    df_c = df.copy()
    df_c.drop(columns=[env.EPISODIC_REWARD_SEED_COL], inplace=True)
    if mode == 1:
        df_c.drop(columns=[env.EPISODIC_REWARD_AVGWINDOW_COL], inplace=True)
    grouped_df = df_c.groupby([env.EPISODIC_REWARD_COLNAME, env.EPISODIC_REWARD_PLATOON_COL])
    avg_df = grouped_df.agg([np.mean, np.std])
    util.write_csv_from_df(avg_df,os.path.join(accum_path, get_mode_tag(mode,AVERAGED_DF)))
    return avg_df

def plot(df, n_vehicles: int, mode: int, accum_path: str):
    n_platoons = len(set(df.index.get_level_values(1)))

    for p_idx in range(n_platoons):
        fig, ax = plt.subplots()
        sub_df = df.xs(p_idx+1, level=env.EPISODIC_REWARD_PLATOON_COL) # Select out data pertinent to the platoon idx, p_idx
        # sub_df = df.loc[(slice(None), p_idx+1), :] # same as above but with slicers
        v_colname = "Vehicle"
        sub_df.columns.names = [v_colname, None]
        sub_df = sub_df.stack(v_colname).reset_index().sort_values(env.EPISODIC_REWARD_COLNAME).reset_index(drop=True)
        for i, v in sub_df.groupby(v_colname):
            ax.plot(v[env.EPISODIC_REWARD_COLNAME], v['mean'], linewidth=get_plot_weight(mode), label=v[v_colname].unique()[0])
            ax.fill_between(v[env.EPISODIC_REWARD_COLNAME],
                            v['mean'] - v['std'], 
                            v['mean'] + v['std'], alpha=0.35)
            ax.set_xlabel("Training episode")
            ax.set_ylabel(get_y_axis_title(mode))
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(accum_path, get_mode_tag(mode,f"pl{p_idx+1}reward.svg")), dpi=150)

def get_vehicle_colnames(n_vehicles: int):
    return [env.EPISODIC_REWARD_VEHICLE_COL_TEMPL % (v_idx) for v_idx in range(1, n_vehicles+1)]

def get_agg_fname(mode) -> bool:
    return AGG_FNAME if mode == 0 else AGG_AVG_FNAME

def get_mode_tag(mode, s) -> str:
    if mode == 0:
        return "ep_" + s
    elif mode == 1:
        return "avgep_" + s

def get_y_axis_title(mode) -> str:
    if mode == 0:
        return "Average episodic reward"
    elif mode == 1:
        return "Cumulative average episodic reward"

def get_plot_weight(mode) -> int:
    if mode == 0:
        return 0.35
    elif mode == 1:
        return 0.5