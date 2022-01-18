import tensorflow as tf
import numpy as np
from workers import trainer, controller, evaluator, accumulator
from src import config, util, reporter, rand
from src.cmd import api
import os 
import random
import sys
import logging
import datetime


logger = logging.getLogger(__name__)

def setup_global_logging_stream(conf):
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(conf.log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def run(args):
    conf = config.Config()
    args, conf = api.get_cmdl_args(args[1:], "Autonomous Vehicle Platoon with DDPG.", conf)

    # set the seed for everything
    rand.set_global_seed(conf.random_seed)

    physical_devices = tf.config.list_physical_devices('GPU') 

    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    util.inititialize_dirs(conf)

    root_dir = sys.path[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    if args.mode == 'tr':
        base_dir = os.path.join(sys.path[0], conf.res_dir, timestamp+f"_{conf.model}_seed{conf.random_seed}_{conf.framework}_{conf.fed_method}")
        os.mkdir(base_dir)

        """ Setup logging to file and console """
        logging.basicConfig(level=logging.INFO,
                            format=conf.log_format,
                            datefmt=conf.log_date_fmt,
                            filename=os.path.join(base_dir, "out.log"),
                            filemode='w')

        setup_global_logging_stream(conf)
        tr = trainer.Trainer(base_dir, timestamp, debug_enabled=args.tr_debug, conf=conf)
        tr.initialize()
        tr.run()
        
    elif args.mode == 'pid':
        controller.run()

    elif args.mode == 'esim': # run eval with that of conf.json
        setup_global_logging_stream(conf)
        conf_path = os.path.join(args.exp_path, config.Config.param_path)
        logger.info(f"Loading configuration instance from {conf_path}")
        conf = util.config_loader(conf_path)
        for p in range(conf.num_platoons):
            evaluator.run(conf=conf, root_path=args.exp_path, out='save', seed=True, pl_idx=p+1, debug_enabled=args.sim_debug, render=args.sim_render) # already seeded above

    elif args.mode == "accumr":
        setup_global_logging_stream(conf)
        accumulator.generate_reward_plot(mode=args.acc_avg, n_vehicles=args.acc_nv, timestamp=timestamp)


    elif args.mode == 'lsim':
        setup_global_logging_stream(conf)
        util.print_dct(util.load_json(args.config_path))

    elif args.mode == 'lmany':
        setup_global_logging_stream(conf)
        report_root = os.path.join(root_dir, conf.report_dir)
        res_dir = os.path.join(root_dir, conf.res_dir)
        list_of_exp_paths = util.find_files(os.path.join(res_dir, '*'))
        
        reporter.generate_latex_report(report_root, list_of_exp_paths, conf.param_path, conf.index_col,
                            conf.drop_keys_in_report, timestamp, 0.5, conf.param_descs)
        

if __name__ == "__main__":
    run(sys.argv)

