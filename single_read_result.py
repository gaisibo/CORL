import re
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


if __name__ == '__main__':
    if not os.path.exists("single_result_files"):
        os.makedirs("single_result_files")
    np.set_printoptions(precision=2)
    # algos = ['iql', 'iqln_10', 'iqln_100', 'iqln2_10', 'iqln2_100', 'iqln3_10', 'iqln3_100', 'sacn_10', 'sacn_100', 'edac_10', 'edac_100', 'td3_plus_bc', 'cql']
    algos = ['iql', 'iqln_100', 'sacn_100', 'edac_10', 'td3_plus_bc']
    # colors = ['#226E9C', '#045275', '#7CCBA2', '#FF1F5B', '#E9002D', '#00CD6C', '#00B000', '#228B3B', '#9CCEA7', '#C40F5B', '#EB85B0', '#F08F6E', '#B10026']
    colors = ['#226E9C', '#045275', '#FF1F5B', '#E9002D', '#00CD6C']
    colors = dict(zip(algos, colors))
    weight_temps = ["3.0", "10.0"]
    expectiles = ["0.7", "0.9", "0.99", "0.995", "0.999"]
    # replay_types = ['bc', 'ewc', 'si', 'gem', 'agem']
    experiences = ['q']
    datasets = ["halfcheetah-random-v0", "hopper-random-v0", "walker2d-random-v0"]
    task_nums = ["0_0_0_300_300_300_0_0_0", "0_0_0_300_300_300_0_0_0", "0_0_0_300_300_300_0_0_0"]
    # max_save_nums = ['1000', '10000', '100000']
    clone_actors = ['', 'cloneactor']
    max_save_nums = ['0', '5', '10', '50', '75', '100']
    entropy_times = ['0', '0.2']
    seeds = ['0']
    continual_types = ["orl_1_orl_1", 'si_1_si_1', 'ewc_1_ewc_1', 'rwalk_1_rwalk_1', 'agem_1_agem_1']
    if not os.path.isdir('single_output_files'):
        raise NotImplementedError
    else:
        file_list = os.listdir('single_output_files')
        # assert False
    results = dict()
    for algo, weight_temp, expectile, experience, continual_type, (dataset, task_num), max_save_num, entropy_time, clone_actor in product(algos, weight_temps, expectiles, experiences, continual_types, zip(datasets, task_nums), max_save_nums, entropy_times, clone_actors):
        if 'iqln' not in algo and 'sacn' not in algo and 'edac' not in algo:
            algo_file = algo + '_10'
        else:
            algo_file = algo
        item_name = algo_file + '_' + weight_temp + '_' + expectile + '_0.7_0.7_' + dataset + '_' + task_num + '_50000_' + continual_type + '_' + entropy_time + '_' + clone_actor + '_' + experience + '_' + max_save_num
        item_name_split = algo + ';' + weight_temp + ';' + expectile + ';' + dataset + ';' + task_num + ';' + continual_type + ';' + clone_actor + ';' + experience + ';' + max_save_num
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
                continue
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
            results[item_name_split] = (mean_last, mean_acc, mean_bwt, var_last, var_acc, var_bwt, real_accs)
    with open(f'single_result_files/results_summary.txt', 'w') as fw:
        for dataset in datasets:
            print(f"{dataset=}", file=fw)
            print("ALL: ", file=fw)
            print("\tLAST:", file=fw)
            for key, value in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
                if dataset in key:
                    print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
            print("", file=fw)
            print("\tACC:", file=fw)
            for key, value in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
                if dataset in key:
                    print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
            print("", file=fw)
            print("\tBWT:", file=fw)
            for key, value in sorted(results.items(), key=lambda x: x[1][2], reverse=True):
                if dataset in key:
                    print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
            print("", file=fw)

            for max_save_num in max_save_nums:
                print(f"{max_save_num=}: ", file=fw)
                print("\tLAST:", file=fw)
                for key, value in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
                    if dataset in key and key.split(";")[-1] == max_save_num:
                        print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
                print("", file=fw)
                print("\tACC:", file=fw)
                for key, value in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
                    if dataset in key and key.split(";")[-1] == max_save_num:
                        print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
                print("", file=fw)
                print("\tBWT:", file=fw)
                for key, value in sorted(results.items(), key=lambda x: x[1][2], reverse=True):
                    if dataset in key and key.split(";")[-1] == max_save_num:
                        print(f"\t{key}: {value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}", file=fw)
                print("", file=fw)

    for dataset in datasets:
        # algos = ['iql', 'iqln_100', 'sacn_100', 'edac_100', 'td3_plus_bc', 'cql']
        algos = ['iql', 'iqln_100', 'sacn_100', 'edac_10', 'td3_plus_bc']
        for max_save_num in max_save_nums:
            if max_save_num == '0':
                continual_type = 'ewc_10.0_ewc_1'
            else:
                continual_type = 'orl_1_orl_1'
            plt.figure()
            plt.cla()
            plt.rc("legend")
            for algo in algos:
                # if algo in ['sacn_100', 'edac_100', 'td3_plus_bc', 'cql']:
                if algo in ['edac_10']:
                    expectile = 0.9
                elif algo in ['sacn_100']:
                    expectile = 0.7
                elif algo in [ 'td3_plus_bc']:
                    expectile = 0.7
                elif algo == 'iql':
                    expectile = 0.9
                else:
                    expectile = 0.99
                key = algo + ';' + '3.0' + ';' + str(expectile) + ';' + dataset + ';' + "0_0_0_300_300_300_0_0_0" + ';' + continual_type + ';' + 'cloneactor' + ';' + 'q' + ';' + max_save_num
                if key not in results.keys():
                    print(f"{key=}")
                    # may_keys = [result for result in results.keys() if algo == result.split(';')[0] and max_save_num == result.split(';')[-1]]
                    may_keys = [result for result in results.keys() if algo == result.split(';')[0]]
                    print(f"{may_keys=}")
                if key in results.keys() and key.split(";")[-1] == max_save_num:
                    algo = key.split(";")[0]
                    real_acc = results[key][-1]
                    x = np.arange(len(real_acc))
                    plt.plot(x, real_acc, label=algo.split('_')[0], c=colors[algo])
            plt.legend(loc="lower left")
            plt.xlabel("Train Process")
            plt.ylabel("Reward")
            # save_path = f"pictures/result_{max_save_num}.png"
            save_path = f"pictures/result_{dataset}_{max_save_num}.png"
            print(f"save to {save_path}")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    for dataset in datasets:
        # algos = ['iql', 'iqln_100', 'sacn_100', 'edac_100', 'td3_plus_bc', 'cql']
        algo = []
        for continual_type in continual_types:
            plt.figure()
            plt.cla()
            plt.rc("legend")
            for algo in algos:
                # if algo in ['sacn_100', 'edac_100', 'td3_plus_bc', 'cql']:
                if algo in ['edac_10']:
                    expectile = 0.9
                elif algo in ['sacn_100']:
                    expectile = 0.99
                elif algo in [ 'td3_plus_bc']:
                    expectile = 0.7
                elif algo == 'iql':
                    expectile = 0.9
                else:
                    expectile = 0.99
                key = algo + ';' + '3.0' + ';' + str(expectile) + ';' + dataset + ';' + "0_0_0_300_300_300_0_0_0" + ';' + continual_type + ';' + 'cloneactor' + ';' + 'q' + ';' + max_save_num
                if key not in results.keys():
                    print(f"{key=}")
                    # may_keys = [result for result in results.keys() if algo == result.split(';')[0] and max_save_num == result.split(';')[-1]]
                    may_keys = [result for result in results.keys() if algo == result.split(';')[0]]
                    print(f"{may_keys=}")
                if key in results.keys() and key.split(";")[-1] == max_save_num:
                    algo = key.split(";")[0]
                    real_acc = results[key][-1]
                    x = np.arange(len(real_acc))
                    plt.plot(x, real_acc, label=algo.split('_')[0], c=colors[algo])
            plt.legend(loc="lower left")
            plt.xlabel("Train Process")
            plt.ylabel("Reward")
            # save_path = f"pictures/result_{max_save_num}.png"
            save_path = f"pictures/result_{dataset}_{max_save_num}.png"
            print(f"save to {save_path}")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
