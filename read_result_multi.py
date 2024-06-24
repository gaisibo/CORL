import re
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    algos = ['td3_plus_bc']
    # replay_types = ['bc', 'ewc', 'si', 'gem', 'agem']
    replay_types = ['bc']
    experiences = ['coverage']
    experiences_long = ['coverage_no_0_none']
    datasets = ["ant_dir_medium_random", "ant_dir_medium", "walker_dir_medium_random", "walker_dir_medium", "cheetah_dir_medium_random", "cheetah_dir_medium", "cheetah_vel_medium_random", "cheetah_vel_medium"]
    task_nums = ["5", "5", '5', '5', "2", "2", '4', '4']
    # max_save_nums = ['1000', '10000', '100000']
    # clone_actors = ['_clone', '_clone_finish', '']
    replay_alphas = ['1']
    max_save_nums = ['1000']
    clone_actors = ['_clone']
    seeds = ['0']
    if not os.path.isdir('output_files'):
        raise NotImplementedError
    else:
        file_list = os.listdir('output_files')
        # print(file_list)
        # assert False
    results = dict()
    for algo, replay_type, (experience, experience_long), (dataset, task_num), max_save_num, alpha, clone_actor in product(algos, replay_types, zip(experiences, experiences_long), zip(datasets, task_nums), max_save_nums, replay_alphas, clone_actors):
        save_name = 'result_' + algo + '_' + replay_type + '_' + experience_long + '_' + dataset + '_' + alpha + '_' + max_save_num + clone_actor
        print(save_name)
        mean_accs, mean_bwts = [], []
        for seed in seeds:
            print(f'seed: {seed}')
            task_num = int(task_num)
            file_model = 'output_' + algo + '_' + replay_type + '_' + experience_long + '_' + dataset + '_' + alpha + '_' + max_save_num + clone_actor + '_' + seed
            file_time = -1
            file_name_match = None
            for file_name in file_list:
                if re.match(file_model, file_name) is not None:
                    try:
                        file_time_new = int(file_name[-18:-4])
                    except ValueError as e:
                        # 没有后缀的是旧版的
                        file_time_new = 0
                    if file_time < file_time_new:
                        file_time = file_time_new
                        file_name_match = file_name
            if file_name_match is None:
                continue
                print(f'file {file_model} not match')
            file_name = 'output_files/' + file_name_match
            real_envs = [[] for _ in range(task_num)]
            if not os.path.isfile(file_name):
                continue
                print(f'file {file_name} not exist')
            with open(file_name, 'r') as fr:
                conti = False
                while True:
                    line = fr.readline()
                    if line == '':
                        break
                    for task_id in range(task_num):
                        try:
                            epoch_num = 30
                            if re.search('epoch=' + str(epoch_num), line) is not None:
                                pattern = re.compile(r"'real_env" + str(task_id) + "': " + r"[-+]?[0-9]*\.?[0-9]+")
                                match = re.search(pattern, line)
                                if match is not None:
                                    match_str = line[match.start() + 13: match.end()]
                                    real_envs[task_id].append(float(match_str))
                        except:
                            conti = True
                            break
                if conti:
                    continue
            macs = []
            if len(real_envs[0]) < task_num:
                continue
            elif len(real_envs[0]) > task_num:
                for i, real_env in enumerate(real_envs):
                    real_envs[i] = real_env[:-1]
            for task_id in range(task_num):
                macs.append(real_envs[task_id][0])
            bwts = []
            accs = []
            for mac, end in zip(macs, [real_env[-1] for real_env in real_envs]):
                accs.append(end)
                bwts.append(mac - end)
            mean_acc = sum(accs) / len(accs)
            mean_accs.append(mean_acc)
            mean_bwt = sum(bwts) / len(bwts)
            mean_bwts.append(mean_bwt)
            with open(f'result_files/result{file_name_match[6:]}.txt', 'w') as fw:
                print(f'real_envs: {real_envs}', file=fw)
        mean_accs = np.array(mean_accs)
        mean_bwts = np.array(mean_bwts)
        mean_acc = mean_accs.mean()
        mean_bwt = mean_bwts.mean()
        var_acc = mean_accs.var()
        var_bwt = mean_bwts.var()
        with open(f'result_files/{save_name}.txt', 'w') as fw:
            print(f'mean_acc: {mean_acc}', file=fw)
            print(f'mean_bwt: {mean_bwt}', file=fw)
            print(f'var_acc: {var_acc}', file=fw)
            print(f'var_bwt: {var_bwt}', file=fw)
        with open(f'result_files/results.txt', 'a') as fw:
            print(f'save_name: {save_name}', file=fw)
            print(f'mean_acc: {mean_acc}', file=fw)
            print(f'mean_bwt: {mean_bwt}', file=fw)
            print(f'var_acc: {var_acc}', file=fw)
            print(f'var_bwt: {var_bwt}', file=fw)
            print('', file=fw)
        results[algo + '_' + replay_type + '_' + experience_long + '_' + dataset + '_' + alpha + '_' + max_save_num + clone_actor] = ("{:.2f}".format(mean_acc), "{:.2f}".format(mean_bwt), "{:.2f}".format(var_acc), "{:.2f}".format(var_bwt))
    print(results)
    with open(f'result_files/result_table_alpha.txt', 'a') as fw:
        print(r'\hline', file=fw)
        print(r'&\multicolumn{2}{|c|}{Ant Dir}&\multicolumn{2}{|c|}{Walker Dir}&\multicolumn{2}{|c|}{Cheetah Dir}&\multicolumn{2}{|c|}{Cheetah Vel}\\', file=fw)
        print(r'\cline{2-9}', file=fw)
        print(r'&Acc&BWT&Acc&BWT&Acc&BWT&Acc&BWT\\', file=fw)
        print(r'\hline', file=fw)
        for experience_long in experiences_long:
            for clone in clone_actors:
                for alpha in replay_alphas:
                    ant_dir_result = results.get("td3_plus_bc_bc_" + experience_long + "_ant_dir_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
                    walker_dir_result = results.get("td3_plus_bc_bc_" + experience_long + "_walker_dir_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
                    cheetah_dir_result = results.get("td3_plus_bc_bc_" + experience_long + "_cheetah_dir_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
                    cheetah_vel_result = results.get("td3_plus_bc_bc_" + experience_long + "_cheetah_vel_medium_" + alpha + "_1000" + clone, ("", "", "", ""))
                    print(fr'{experience_long}{clone}&{ant_dir_result[0]}&{ant_dir_result[1]}&{walker_dir_result[0]}&{walker_dir_result[1]}&{cheetah_dir_result[0]}&{cheetah_dir_result[1]}&{cheetah_vel_result[0]}&{cheetah_vel_result[1]}\\', file=fw)
        print(r'\\', file=fw)
        print(r'\hline', file=fw)
