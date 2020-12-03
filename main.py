import pandas as pd
import pickle

from models import LinearModel, RNN
from utils.cachedir import cache_dir

def main():
    # LinearModel.load_data() # comment this out to use cache
    # train_data = pickle.load(open(f'{cache_dir}/train_embedded.p', 'rb'))
    # test_data = pickle.load(open(f'{cache_dir}/test_embedded.p', 'rb'))
    # print('done loading')

    RNN.load_data()
    train_data = pickle.load(open(f'{cache_dir}/train_rnn.p', 'rb'))
    test_data = pickle.load(open(f'{cache_dir}/test_rnn.p', 'rb'))
    print('done loading')
    model = RNN(train_data, test_data)
    model.train()
    model.test()




if __name__ == '__main__':
    main()
