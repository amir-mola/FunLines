import pickle
from models import LinearModel
from scripts.datadir import data_dir

def main():
    LinearModel.load_data()  # comment this out to re-use last split
    train_data = pickle.load(open(f'{data_dir}/train_embed.p', 'rb'))
    test_data = pickle.load(open(f'{data_dir}/test_embed.p', 'rb'))
    print('done loading')

    model = LinearModel(train_data, test_data)
    model.train()
    model.test()


if __name__ == '__main__':
    main()
