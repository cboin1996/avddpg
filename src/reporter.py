from src import util
import pandas as pd
import os
import shutil

def aggregate_json_to_df(output_root, list_of_exp_paths, conf_fname, report_timestamp, drop_cols=None, index_col=None, save_latex_out=True):
    """[summary]
    Args:
        output_root (str): The string of the rroto reporting folder
        list_of_exp_paths (list): List of experiments (directories) to scrape info from
        conf_name (str): Name of json with hyperparameters
        report_timestamp (str): timestampe for the report folder
    """
    report_dir = os.path.join(output_root, report_timestamp)
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
    for i, dname in enumerate(list_of_exp_paths):
        conf_fpath = os.path.join(dname, conf_fname)
        conf_dct = util.load_json(conf_fpath)
        conf_dct = util.remove_keys_from_dict(conf_dct, drop_cols)
        if i == 0:
            aggregate_df = pd.DataFrame([conf_dct])
        else:
            aggregate_df = aggregate_df.append(pd.DataFrame([conf_dct]))
    
    if index_col is not None:
        aggregate_df = aggregate_df.set_index(index_col)
    util.write_csv_from_df(aggregate_df, os.path.join(report_dir, report_timestamp + ".csv"))
    if save_latex_out:
        util.save_file(os.path.join(report_dir, report_timestamp + ".txt"), aggregate_df.to_latex())
    else:
        return aggregate_df

def get_figure_str(fig_width, fig_path, fig_label, fig_caption):
    figure =    """
        \\begin{figure}
        \caption{%s}
        \centering
            \includegraphics[width=%s\linewidth]{%s}
        \label{%s}
        \end{figure}
        """ % (fig_caption, fig_width, fig_path, fig_label)
    return figure

def get_svg_str(fig_width, fig_path, fig_label, fig_caption):
    figure =    """
        \\begin{figure}
        \caption{%s}
        \centering
            \includesvg[width=%s\linewidth]{%s}
        \label{%s}
        \end{figure}
        """ % (fig_caption, fig_width, fig_path, fig_label)
    return figure

def generate_fig_params(experiment_dir, conf_fname):
    conf = util.config_loader(os.path.join(experiment_dir, conf_fname))
    fig_params = [{"name" : conf.actor_picname % (1, 1),
            "width" : 0.5,
            "caption" : "Actor network model for experiment %s"},
            {"name" : conf.critic_picname % (1, 1),
            "width" : 0.6,
            "caption" : "Critic network model for experiment %s"}
    ]

    for p in range(conf.num_platoons):
        fig_params.append( {"name" : conf.fig_path % (p+1),
                "width" : 0.6,
                "caption" : f"Platoon {p+1} reward curve for experiment %s"}
                )
        fig_params.append({"name" : f"res_guassian{conf.pl_tag}.svg" % (p+1),
                "width" : 0.4,
                "caption" : f"Platoon {p+1} simulation for experiment %s"})
    
    return fig_params

def generate_latex_report(output_root, list_of_exp_paths, conf_fname, conf_index_col, conf_drop_cols, report_timestamp, fig_width, param_dct):
    """Generates a latex report body

    Args:
        output_root (str): The string of the rroto reporting folder
        list_of_exp_paths (str): List of experiments (directories) to scrape info from
        conf_fname (str): Name of json with hyperparameters
        conf_index_col (str) : the config files key for setting index of dataframe
        conf_drop_cols (str) : any columns to drop from the dataframe
        report_timestamp (str): timestampe for the report folder
        fig_names (list): file names of any figures to include in the report
        fig_width (float): width of the figures
        json_names (str): json for hyperparameter descriptions
    """
    report_dir = os.path.join(output_root, report_timestamp)
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
    report_txt_fname = 'latex_results.tex'
    with open(os.path.join(report_dir, report_txt_fname), 'a') as f:
        f.write('\section{Parameter Descriptions}\n')
        f.write(pd.DataFrame({'Hyperparameter' : param_dct.keys(), "Description" : param_dct.values()}).to_latex(caption="Hyperparameter Legend", label="tab:hyplegend"))

        f.write('\section{Summary of Results}')
        all_confs_df = aggregate_json_to_df(output_root,
                                    list_of_exp_paths,
                                    conf_fname,
                                    report_timestamp,
                                    drop_cols=conf_drop_cols,
                                    index_col=conf_index_col,
                                    save_latex_out=False)
        f.write(all_confs_df.to_latex(caption=f"Results Across All Experiments with Hyperparameters", label=f"tab:all_hyps"))

        # get columns that are constant from the dataframe, to make a summary table of the constant params with their values
        constant_value_cols = [c for c in all_confs_df.columns if len(set(all_confs_df[c])) == 1]
        all_confs_df_constants = all_confs_df[constant_value_cols][:1].T.reset_index()
        all_confs_df_constants.columns = ['Hyperparameters', 'Value']
        util.write_csv_from_df(all_confs_df_constants, os.path.join(report_dir, "constants.csv"))
        f.write(all_confs_df_constants.to_latex(caption=f"Hyperparameters Unchanged Across All Experiments", label=f"tab:const_hyps"))

        # make a table of those that changed
        all_confs_df_changed = all_confs_df.drop(columns=constant_value_cols)
        util.write_csv_from_df(all_confs_df_changed, os.path.join(report_dir, "changed.csv"))
        f.write(all_confs_df_changed.to_latex(caption=f"Hyperparameters That Changed Across All Experiments", label=f"tab:changed_hyps"))

        for dir_ in list_of_exp_paths:
            dir_name = os.path.basename(os.path.normpath(dir_))
            latexify_dir_name = util.latexify(dir_name)
            f.write('\section{Experiment %s}\n' % (latexify_dir_name))
            report_exp_dir = os.path.join(report_dir, dir_name)
            os.mkdir(report_exp_dir)

            fig_params = generate_fig_params(dir_, conf_fname)

            for fig_map in fig_params:
                fig_name = fig_map['name']
                fig_src = os.path.join(dir_, fig_name)
                fig_dest = os.path.join(report_exp_dir, fig_name)
                fig_relative_path = dir_name + "/" + fig_name
                shutil.copy(fig_src, fig_dest)
                if '.png' in fig_name:
                    f.write(get_figure_str(fig_map['width'], fig_relative_path, f"fig:{fig_relative_path}", fig_map["caption"] % (latexify_dir_name)))
                elif '.svg' in fig_name:
                    f.write(get_svg_str(fig_map['width'], fig_relative_path, f"fig:{fig_relative_path}", fig_map["caption"] % (latexify_dir_name)))
                else:
                    raise ValueError(f"Cannot load image. Consider converting image {fig_src} to one of '.png' or '.svg' and try running again.")

            conf_fpath = os.path.join(dir_, conf_fname)
            conf_dct = util.load_json(conf_fpath)
            conf_dct = util.remove_keys_from_dict(conf_dct, conf_drop_cols)

            conf_df = pd.DataFrame({'Hyperparameter' : conf_dct.keys(), "Values" : conf_dct.values()})
            f.write(conf_df.to_latex(caption=f"Hyperparameter's for Experiment {latexify_dir_name}", label=f"tab:hyp_exp{dir_name}"))


        

            

                

    