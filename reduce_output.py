import os
import shutil


if __name__ == '__main__':
    dir_path = './single_output_files'
    for file_name in os.listdir(dir_path):
        if 'antmaze' not in file_name:
            file_path = os.path.join(dir_path, file_name)
            file_rename_path = os.path.join(dir_path, file_name + '.rename')
            i = 0
            have = False
            with open(file_path, 'r') as fr:
                line = fr.readline()
                while line is not None:
                    if line.startswith('self._critic_update_step') or line.startswith('total_step') or line.startswith('coldstart_step'):
                        have = True
                        line = fr.readline()
                        break
                    line = fr.readline()
                    i += 1
                    if i > 1000:
                        break
            if have:
                with open(file_path, 'r') as fr:
                    with open(file_rename_path, 'w') as fw:
                        line = fr.readline()
                        while line is not None and line != '':
                            if not (line.startswith('self._critic_update_step') or line.startswith('total_step') or line.startswith('coldstart_step') or line.startswith('Target') or line.startswith('len')):
                                print(line, file=fw, end='')
                            line = fr.readline()
                shutil.move(file_rename_path, file_path)
