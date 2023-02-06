import pickle


if __name__ == '__main__':
    for i in range(50):
        with open(f'env_ant_dir_train_task{i}.pkl', 'rb') as f:
            x = pickle.load(f)
        print(f"x: {x}")
