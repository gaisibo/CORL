import re
import os
import copy
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fontsize={'family': 'DejaVu Sans Mono', 'weight': 'normal', 'size': 20,}
    plt.rc('font', **fontsize)
    files = dict()
    files["embed_orl_1_1e-3_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-3_none_1_3e-4_0__random_1_--embed_0_20240307134000.txt"
    files["embed_orl_1_1e-4_1e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-4_none_1_3e-4_0__random_1_--embed_0_20240307134000.txt"
    files["embed_orl_1_1e-4_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-4_none_1_1e-4_0__random_1_--embed_0_20240307134000.txt"
    files["embed_orl_1_1e-5_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-5_none_1_3e-4_0__random_1_--embed_0_20240307134000.txt"
    files["embed_none_1e-3_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_none_1_1e-4_none_1_3e-4_0__random_1_--embed_0_20240307134003.txt"
    files["embed_none_1e-4_1e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_none_1_1e-4_none_1_3e-4_0__random_1_--embed_0_20240307134003.txt"
    files["embed_none_1e-4_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_none_1_1e-4_none_1_3e-4_0__random_1_--embed_0_20240307134003.txt"
    files["embed_none_1e-5_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_none_1_1e-4_none_1_3e-4_0__random_1_--embed_0_20240307134003.txt"
    files["embed_orl_10_1e-3_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-3_none_1_3e-4_0__random_10_--embed_0_20240305015723.txt"
    files["embed_orl_10_1e-4_1e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-4_none_1_3e-4_0__random_10_--embed_0_20240307134102.txt"
    files["embed_orl_10_1e-4_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-4_none_1_1e-4_0__random_10_--embed_0_20240307134102.txt"
    files["embed_orl_10_1e-5_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-5_none_1_3e-4_0__random_10_--embed_0_20240307134102.txt"
    files["embed_orl_3_1e-3_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-3_none_1_3e-4_0__random_10_--embed_0_20240307134227.txt"
    files["embed_orl_3_1e-4_1e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-4_none_1_3e-4_0__random_10_--embed_0_20240307134227.txt"
    files["embed_orl_3_1e-4_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-4_none_1_1e-4_0__random_10_--embed_0_20240307134227.txt"
    files["embed_orl_3_1e-5_3e-4"] = "few_shot_output_files/output_iql_10_3.0_0.9_0.5_0.5_halfcheetah_3000-0_10000_orl_1_1e-5_none_1_3e-4_0__random_10_--embed_0_20240307134227.txt"

    outer_weight_all = dict()
    outer_log_prob_all = dict()
    outer_actor_loss_all = dict()
    testing_weight_all = dict()
    testing_log_prob_all = dict()
    testing_actor_loss_all = dict()
    for key, value in files.items():
        output_name, eval_name = value, value

        outer_weight_all[key] = []
        outer_weight = outer_weight_all[key]
        outer_log_prob_all[key] = []
        outer_log_prob = outer_log_prob_all[key]
        outer_actor_loss_all[key] = []
        outer_actor_loss = outer_actor_loss_all[key]

        pattern_weight = re.compile(r"'outer_weight': " + r"[-+]?[0-9]*\.?[0-9]+")
        pattern_log_prob = re.compile(r"'outer_log_prob': " + r"[-+]?[0-9]*\.?[0-9]+")
        pattern_actor_loss = re.compile(r"'outer_actor_loss': " + r"[-+]?[0-9]*\.?[0-9]+")
        testing = False
        replaying = False
        with open(output_name, 'r') as output_file:
            line = output_file.readline()
            while line:
                if "Testing" in line:
                    testing = True
                elif "Start Replaying" in line:
                    replaying = True
                    line = output_file.readline()
                    line = output_file.readline()
                    line = output_file.readline()
                elif "Start" in line and "Start dataset" not in line:
                    replaying = False
                    testing = False
                if not testing and not replaying:
                    match_weight = re.search(pattern_weight, line)
                    if match_weight is not None:
                        match_weight = float(line[match_weight.start() + 16: match_weight.end()])
                        outer_weight.append(min(match_weight, 1))

                    match_log_prob = re.search(pattern_log_prob, line)
                    if match_log_prob is not None:
                        match_log_prob = float(line[match_log_prob.start() + 18: match_log_prob.end()])
                        outer_log_prob.append(max(match_log_prob, -1))

                    match_actor_loss = re.search(pattern_actor_loss, line)
                    if match_actor_loss is not None:
                        match_actor_loss = float(line[match_actor_loss.start() + 20: match_actor_loss.end()])
                        outer_actor_loss.append(min(match_actor_loss, 10))
                line = output_file.readline()
        with open(eval_name, 'r') as eval_file:
            line = eval_file.readline()
            while line:
                if "Testing" in line and "Testing after" not in line:
                    if "ant_dir" in line:
                        testing_weight_dir["ant_dir"] = []
                        testing_weight = testing_weight_dir["ant_dir"]
                        testing_log_prob_dir["ant_dir"] = []
                        testing_log_prob = testing_log_prob_dir["ant_dir"]
                        testing_actor_loss_dir["ant_dir"] = []
                        testing_actor_loss = testing_actor_loss_dir["ant_dir"]
                    elif "cheetah_vel" in line:
                        testing_weight_dir["cheetah_vel"] = []
                        testing_weight = testing_weight_dir["cheetah_vel"]
                        testing_log_prob_dir["cheetah_vel"] = []
                        testing_log_prob = testing_log_prob_dir["cheetah_vel"]
                        testing_actor_loss_dir["cheetah_vel"] = []
                        testing_actor_loss = testing_actor_loss_dir["cheetah_vel"]
                    elif "walker_dir" in line:
                        testing_weight_dir["walker_dir"] = []
                        testing_weight = testing_weight_dir["walker_dir"]
                        testing_log_prob_dir["walker_dir"] = []
                        testing_log_prob = testing_log_prob_dir["walker_dir"]
                        testing_actor_loss_dir["walker_dir"] = []
                        testing_actor_loss = testing_actor_loss_dir["walker_dir"]

                    match_weight = re.search(pattern_weight, line)
                    if match_weight is not None:
                        match_weight = float(line[match_weight.start() + 16: match_weight.end()])
                        testing_weight.append(min(match_weight, 1))

                    match_log_prob = re.search(pattern_log_prob, line)
                    if match_log_prob is not None:
                        match_log_prob = float(line[match_log_prob.start() + 18: match_log_prob.end()])
                        testing_log_prob.append(max(match_log_prob, -1))

                    match_actor_loss = re.search(pattern_actor_loss, line)
                    if match_actor_loss is not None:
                        match_actor_loss = float(line[match_actor_loss.start() + 20: match_actor_loss.end()])
                        testing_actor_loss.append(min(match_actor_loss, 10))
                line = eval_file.readline()

        testing_weight_dir["ant_dir"] = sum(testing_weight_dir["ant_dir"]) / len(testing_weight_dir["ant_dir"])
        testing_weight_dir["cheetah_vel"] = sum(testing_weight_dir["cheetah_vel"]) / len(testing_weight_dir["cheetah_vel"])
        testing_weight_dir["walker_dir"] = sum(testing_weight_dir["walker_dir"]) / len(testing_weight_dir["walker_dir"])
        testing_log_prob_dir["ant_dir"] = sum(testing_log_prob_dir["ant_dir"]) / len(testing_log_prob_dir["ant_dir"])
        testing_log_prob_dir["cheetah_vel"] = sum(testing_log_prob_dir["cheetah_vel"]) / len(testing_log_prob_dir["cheetah_vel"])
        testing_log_prob_dir["walker_dir"] = sum(testing_log_prob_dir["walker_dir"]) / len(testing_log_prob_dir["walker_dir"])
        testing_actor_loss_dir["ant_dir"] = sum(testing_actor_loss_dir["ant_dir"]) / len(testing_actor_loss_dir["ant_dir"])
        testing_actor_loss_dir["cheetah_vel"] = sum(testing_actor_loss_dir["cheetah_vel"]) / len(testing_actor_loss_dir["cheetah_vel"])
        testing_actor_loss_dir["walker_dir"] = sum(testing_actor_loss_dir["walker_dir"]) / len(testing_actor_loss_dir["walker_dir"])

    for name, outer_metric_all, testing_metric_all in [("weight", outer_weight_all, testing_weight_all), ("log_prob", outer_log_prob_all, testing_log_prob_all), ("actor_loss", outer_actor_loss_all, testing_actor_loss_all)]:
        for embed, replay, color in [("none", "none", "red"), ("none", "orl", "orange"), ("embed", "none", "green"), ("embed", "orl", "navy")]:
            output_name, eval_name = files[embed + '_' + replay][0], files[embed + '_' + replay][1]
            outer_metric = outer_metric_all[embed + '_' + replay]
            testing_metric = testing_metric_all[embed + '_' + replay]

            x = np.arange(len(outer_metric))
            x1 = np.arange(len(outer_metric) // 3)
            x2 = np.arange(len(outer_metric) // 3)
            x3 = np.arange(len(outer_metric) // 3)
            plt.plot(x1, outer_metric[: len(outer_metric) // 3], label=embed + '_' + replay, c=color)
            plt.plot(x2 + len(outer_metric) // 3, outer_metric[len(outer_metric) // 3: len(outer_metric) // 3 * 2], c=color, linestyle='-.')
            plt.plot(x3 + len(outer_metric) // 3 * 2, outer_metric[len(outer_metric) // 3 * 2 : ], c=color, linestyle='--')
            # plt.plot(x, outer_metric, c=color)
            plt.plot(x1, [testing_metric["ant_dir"] for _ in range(len(outer_metric) // 3)], c=color)
            plt.plot(x2 + len(outer_metric) // 3, [testing_metric["cheetah_vel"] for _ in range(len(outer_metric) // 3)], c=color, linestyle='-.')
            plt.plot(x3 + len(outer_metric) // 3 * 2, [testing_metric["walker_dir"] for _ in range(len(outer_metric) // 3)], c=color, linestyle='--')
        plt.legend(loc="upper right")
        plt.xlabel("Train Process")
        if name == 'actor_loss':
            plt.ylabel("Actor Loss")
        elif name == 'weight':
            plt.ylabel("Weight")
        elif name == 'log_prob':
            plt.ylabel("Log Prob")
        save_path = f"pictures/result_few_shot_{name}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
