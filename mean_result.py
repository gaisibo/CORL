import numpy as np
from functools import reduce


with open('result_files/result_table_alpha.txt', 'r') as fr:
    result_dict = dict()
    result_dict_ = dict()
    line = fr.readline()
    while line != '':
        line_parts = line.split('&')
        if len(line_parts) == 1:
            line = fr.readline()
            continue
        if line_parts[0] not in result_dict.keys():
            result_dict[line_parts[0]] = np.array([float(x) for x in line_parts[1:]])
        else:
            result_dict[line_parts[0]] += np.array([float(x) for x in line_parts[1:]])
        line = fr.readline()
    for key, item in result_dict.items():
        print(item)
        result_dict_[key] = item / 5
        print(result_dict_[key])
    with open('result_files/result_table_alpha_mean.txt', 'w') as fw:
        for key, item in result_dict_.items():
            print(key)
            print([str(x) for x in item.tolist()])
            print_line = key + '&' + reduce(lambda x, y: x + '&' + y, [("%.2f" % x) for x in item.tolist()]) + r'\\'
            print(print_line, file=fw)
