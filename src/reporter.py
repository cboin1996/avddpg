from src import util
import pandas as pd
import os

def aggregate_json_to_df(output_root, list_of_conf_paths, report_timestamp, drop_cols=None, index_col=None, convert_to_latex=True):
    report_dir = os.path.join(output_root, report_timestamp)
    os.mkdir(report_dir)
    for i, fname in enumerate(list_of_conf_paths):
        conf_dct = util.load_json(fname)
        conf_dct = util.remove_keys_from_dict(conf_dct, drop_cols)
        if i == 0:
            aggregate_df = pd.DataFrame([conf_dct])
        else:
            aggregate_df = aggregate_df.append(pd.DataFrame([conf_dct]))
    
    if index_col is not None:
        aggregate_df = aggregate_df.set_index(index_col)
    util.write_csv_from_df(aggregate_df, os.path.join(report_dir, report_timestamp + ".csv"))
    if convert_to_latex:
        util.save_file(os.path.join(report_dir, report_timestamp + ".txt"), aggregate_df.to_latex())

def generate_latex_report():
    pass
    