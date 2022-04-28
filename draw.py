import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    for m in ['model']:
        for c in ['bc']:
            for a in ['10', '3', '1', '0.3', '0.1', '0.03']:
                name = './output_' + m + '_' + c + '_hopper_medium_v0_10000_' + a + '_draw.txt'
                if os.path.exists(name):
                    print(f'drawing ' + name)
                    plt.figure()
                    fig_name = './output_' + m + '_' + c + '_hopper_medium_v0_10000_' + a + '_.png'
                    r = []
                    for num in range(4):
                        r.append([])
                    with open(name, 'r') as f:
                        for line in f:
                            i_start = 0
                            for num in range(4):
                                try:
                                    i_start = line.index('real_env' + str(num), i_start)
                                    i_start += 12
                                    try:
                                        i_end = line.index(',', i_start)
                                    except:
                                        i_end = line.index('}', i_start)
                                    i_str = line[i_start: i_end]
                                    r[num].append(float(i_str))
                                except:
                                    pass
                    for i in range(4):
                        x = range(len(r[i]))
                        plt.plot(x, r[i])
                    plt.savefig(fig_name)
