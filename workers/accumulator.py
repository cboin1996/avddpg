from distutils.command import config
from logging import root
import pandas as pd
import numpy as np
from src import config, util
from src.env import env
import os, sys, shutil
import glob

AGG_FNAME = "aggregation.csv"
AGG_AVG_FNAME = "cumavg_aggregation.csv"

def generate_reward_plot(mode: int, n_vehicles: int, timestamp: int) -> bool:
    """Accumulates and plots across multiple seeds, averaging the reward value
    across the seed and plotting with a shaded error bar.

    Args:
        mode (int): A value of 0 or 1 dictates which csv file to use for aggregation.
        0 = not averaged, 1 = averaged
    """
    accum_path = os.path.join(sys.path[0], config.Config.report_dir, timestamp)
    os.mkdir(accum_path)
    df = get_data(mode, accum_path)

    df = transform(df, n_vehicles)
    return True


def get_data(mode: int, accum_path: str) -> pd.DataFrame:
    """Aggregates all csv's into a single csv file and writes to disk. Returns the object in a pd.DataFrame.
    """
    path_lst = glob.glob(os.path.join(sys.path[0], config.Config.res_dir, "*"))
    df = None
    for root_path in path_lst:
        conf_path = os.path.join(root_path, config.Config.param_path)
        dir_name = os.path.basename(os.path.normpath(root_path))
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

    agg_fname = AGG_FNAME if mode == 0 else AGG_AVG_FNAME
    util.write_csv_from_df(df,os.path.join(accum_path, agg_fname))
    return df

def transform(df, n_vehicles: int):
    """Performs a transformation on the given df, producing the averaged result

    Args:
        df (pd.DataFrame): the pandas dataframe
        n_vehicles (int): the number of assumed vehicle columns in the data frame.
    """

    # Get the number of platoons out of the data set.  Will return a dataframe for each platoon, averaged for vehicles aross the seeds.
    n_platoons = len(df[env.EPISODIC_REWARD_PLATOON_COL].unique())

    if n_platoons < 1:
        raise ValueError("Number of platoons must be greater than 1!")
    
    transform_df = None
    for v_idx in n_vehicles:
        v_colname = env.EPISODIC_REWARD_VEHICLE_COL_TEMPL % (v_idx)
        # df[]