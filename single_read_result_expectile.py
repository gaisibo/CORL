import re
import os
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product


if __name__ == '__main__':
    if not os.path.exists("single_result_files"):
        os.makedirs("single_result_files")
    mpl.rcParams['text.usetex'] = True
    np.set_printoptions(precision=2)
    algos = ['iqln_100']
    weight_temps = ["3.0"]
    expectiles = ["0.7", "0.8", "0.9", "0.95", "0.99", "0.995", "0.999", "1.0"]
    # colors = ['#009392', '#39B185', '#9CCB86', '#E9E29C', '#EEB479', '#E88471']
    # colors = ['#226E9C', '#045275', '#7CCBA2', '#FF1F5B', '#E9002D', '#00CD6C', '#00B000', '#228B3B']
    # colors = dict(zip(expectiles, colors))
    # replay_types = ['bc', 'ewc', 'si', 'gem', 'agem']
    experiences = ['q']
    datasets = ["halfcheetah-random-v0"]
    task_nums = ["0_0_0_300_300_300_0_0_0"]
    # max_save_nums = ['1000', '10000', '100000']
    entropy_times = ['0', '0.2']
    clone_actors = ['cloneactor']
    max_save_nums = ['75']
    seeds = ['0']
    continual_types = ["orl_1_orl_1"]
    if not os.path.isdir('single_output_files'):
        raise NotImplementedError
    else:
        file_list = os.listdir('single_output_files')
        # assert False
    results = dict()
    for algo, weight_temp, expectile, experience, continual_type, (dataset, task_num), max_save_num, entropy_time, clone_actor in product(algos, weight_temps, expectiles, experiences, continual_types, zip(datasets, task_nums), max_save_nums, entropy_times, clone_actors):
        item_name = algo + '_' + weight_temp + '_' + expectile + '_0.7_0.7_' + dataset + '_' + task_num + '_50000_' + continual_type + '_' + entropy_time + '_' + clone_actor + '_' + experience + '_' + max_save_num
        save_name = 'result_' + item_name
        # print(save_name)
        mean_lasts, mean_accs, mean_bwts, real_accs = [], [], [], []

        for seed in seeds:
            # print(f'seed: {seed}')
            file_model = 'output_' + item_name + '_' + str(seed)
            file_time = -1
            file_name_match = None
            for file_name in file_list:
                if re.match(file_model, file_name) is not None:
                    file_time_new = int(file_name[-18:-4])
                    if file_time < file_time_new:
                        file_time = file_time_new
                        file_name_match = file_name
            if file_name_match is None:
                print(f"{file_model} not match")
                continue
            if "0.8" in item_name:
                print(f"{file_model} match")
            file_name = 'single_output_files/' + file_name_match
            task_num_split = task_num.split('_')
            task_length = len(task_num_split)
            real_envs = []
            if not os.path.isfile(file_name):
                print(f'file {file_name} not exist')
                continue
            with open(file_name, 'r') as fr:
                conti = False
                while True:
                    line = fr.readline()
                    if line == '':
                        break
                    try:
                        epoch_num = 50
                        if re.search('epoch=' + str(epoch_num), line) is not None:
                            pattern = re.compile(r"'0_environment': " + r"[-+]?[0-9]*\.?[0-9]+")
                            match = re.search(pattern, line)
                            if match is not None:
                                match_str = line[match.start() + 17: match.end()]
                                real_envs.append(float(match_str))
                    except:
                        conti = True
                        break
                if conti:
                    continue
            if len(real_envs) < task_length:
                continue
            bwts = []
            accs = []
            mean_acc = sum(real_envs) / len(real_envs)
            mean_accs.append(mean_acc)
            mean_bwt = max(real_envs) - sum(real_envs) / len(real_envs)
            mean_bwts.append(mean_bwt)
            mean_lasts.append(real_envs[-1])
            real_accs.append(real_envs)
            with open(f'single_result_files/result{file_name_match[6:]}.txt', 'w') as fw:
                print(f'real_envs: {real_envs}', file=fw)
        if len(mean_lasts) > 0:
            real_accs = np.array(real_accs)
            real_accs = [np.mean(np.array([x[i] for x in real_accs])) for i in range(len(real_accs[0]))]

            mean_lasts = np.array(mean_lasts)
            mean_accs = np.array(mean_accs)
            mean_bwts = np.array(mean_bwts)
            mean_last = mean_lasts.mean()
            mean_acc = mean_accs.mean()
            mean_bwt = mean_bwts.mean()
            var_last = mean_last.var()
            var_acc = mean_accs.var()
            var_bwt = mean_bwts.var()
            with open(f'single_result_files/{save_name}.txt', 'w') as fw:
                print(f'mean_last: {mean_last}', file=fw)
                print(f'mean_acc: {mean_acc}', file=fw)
                print(f'mean_bwt: {mean_bwt}', file=fw)
                print(f'var_last: {var_last}', file=fw)
                print(f'var_acc: {var_acc}', file=fw)
                print(f'var_bwt: {var_bwt}', file=fw)
            with open(f'single_result_files/results.txt', 'a') as fw:
                print(f'save_name: {save_name}', file=fw)
                print(f'mean_last: {mean_last}', file=fw)
                print(f'mean_acc: {mean_acc}', file=fw)
                print(f'mean_bwt: {mean_bwt}', file=fw)
                print(f'var_last: {var_last}', file=fw)
                print(f'var_acc: {var_acc}', file=fw)
                print(f'var_bwt: {var_bwt}', file=fw)
                print('', file=fw)
            results[expectile] = (mean_last, mean_acc, mean_bwt, var_last, var_acc, var_bwt, real_accs)
    with open(f'single_result_files/results_summary.txt', 'w') as fw:
        print("ALL: ", file=fw)
        print("\tLAST:", file=fw)
        for key, value in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
            print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
        print("", file=fw)
        print("\tACC:", file=fw)
        for key, value in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
            print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
        print("", file=fw)
        print("\tBWT:", file=fw)
        for key, value in sorted(results.items(), key=lambda x: x[1][2], reverse=True):
            print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
        print("", file=fw)

        for max_save_num in max_save_nums:
            print(f"{max_save_num=}: ", file=fw)
            print("\tLAST:", file=fw)
            for key, value in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
                if key.split(";")[-1] == max_save_num:
                    print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
            print("", file=fw)
            print("\tACC:", file=fw)
            for key, value in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
                if key.split(";")[-1] == max_save_num:
                    print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
            print("", file=fw)
            print("\tBWT:", file=fw)
            for key, value in sorted(results.items(), key=lambda x: x[1][2], reverse=True):
                if key.split(";")[-1] == max_save_num:
                    print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
            print("", file=fw)

    plt.figure()
    plt.cla()
    plt.rc("legend")
    for expectile in expectiles:
        key = expectile
        print(f"{results.keys()=}")
        if key in results.keys():
            real_acc = results[key][-1]
            x = np.arange(len(real_acc))
            # plt.plot(x, real_acc, label="expectile: " + expectile, c=colors[key])
            plt.plot(x, real_acc, label=r"$\tau$: " + expectile)
    plt.legend(loc="lower left")
    plt.xlabel("Train Process")
    plt.ylabel("Reward")
    save_path = f"pictures/result_expectile.png"
    print(f"save to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    # with open(f'single_result_files/result_table_alpha.txt', 'a') as fw:
    #     print(r'\hline', file=fw)
    #     print(r'&\multicolumn{2}{|c|}{Ant Dir}&\multicolumn{2}{|c|}{Walker Dir}&\multicolumn{2}{|c|}{Cheetah Dir}&\multicolumn{2}{|c|}{Cheetah Vel}\\', file=fw)
    #     print(r'\cline{2-9}', file=fw)
    #     print(r'&Acc&BWT&Acc&BWT&Acc&BWT&Acc&BWT\\', file=fw)
    #     print(r'\hline', file=fw)
    #     for experience_long in experiences_long:
    #         for clone in clone_actors:
    #             for alpha in replay_alphas:
    #                 ant_dir_result = results.get("td3_plus_bc_bc_" + experience_long + "_ant_dir_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
    #                 walker_dir_result = results.get("td3_plus_bc_bc_" + experience_long + "_walker_dir_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
    #                 cheetah_dir_result = results.get("td3_plus_bc_bc_" + experience_long + "_cheetah_dir_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
    #                 cheetah_vel_result = results.get("td3_plus_bc_bc_" + experience_long + "_cheetah_vel_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
    #                 print(fr'{experience_long}{clone}&{ant_dir_result[0]}&{ant_dir_result[1]}&{walker_dir_result[0]}&{walker_dir_result[1]}&{cheetah_dir_result[0]}&{cheetah_dir_result[1]}&{cheetah_vel_result[0]}&{cheetah_vel_result[1]}\\', file=fw)
    #     print(r'\\', file=fw)
    #     print(r'\hline', file=fw)
