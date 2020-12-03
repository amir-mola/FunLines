from models import LinearModel
from utils.cachedir import cache_dir

import pickle
import csv

path = 'data/train_funlines.csv'


class Replacer():
    def __init__(self):

        # Load / train model
        train_data = pickle.load(open(f'{cache_dir}/train_embedded.p', 'rb'))
        test_data = pickle.load(open(f'{cache_dir}/test_embedded.p', 'rb'))
        print('done loading')
        self.model = LinearModel(train_data, test_data)
        self.model.train()
        self.model.test()

        # Gather replacements
        self.replacements = set()
        with open(path, encoding="utf8", errors='ignore') as f:
            rows = csv.reader(f)
            rows = list(rows)
            for row in rows[1:100]:
                self.replacements.add(row[2])

    def optimize(self, sentence):
        options = [sentence]
        sen_arr = sentence.split()
        for i in range(len(sen_arr)):
            for rep in self.replacements:
                options.append(replace_word(sen_arr, rep, i))

        max_score = 0
        argmax = None
        i = 0
        for option in options:
            option_score = self.model.score(option)
            print(f'{i}/{len(options)}: {option}\t{option_score}')
            if option_score > max_score:
                max_score = option_score
                argmax = option
            i += 1

        print(argmax, max_score)


def replace_word(arr, new, i):
    arr2 = arr.copy()
    arr2[i] = new
    return ' '.join(arr2)


if __name__ == '__main__':
    r = Replacer()
    r.optimize('The quick brown fox jumped over the lazy dog')
