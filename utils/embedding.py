import torch
import pdb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import csv
import re
import tqdm
import multiprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to(device)

max_length = 42


class FunLineDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(FunLineDataset, self).__init__()
        self.data = data

    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
        return len(self.data)

    def __getitem__(self, idx):

        return list(self.data.loc[idx])

# Creates train and test data in pands dataframe


def create_dataset(path):

    data = []
    with open(path, encoding="utf8", errors='ignore') as f:
        rows = csv.reader(f)
        rows = list(rows)
        for row in rows[1:100]:
            if float(row[4]) == 0:
                continue
            match = re.search(r'<.*>', row[1])
            original = row[1][:match.start()] + row[1][match.start() +
                                                       1:match.end()-2] + row[1][match.end():]
            edited = row[1][:match.start()] + row[2] + row[1][match.end():]
            masked = row[1][:match.start()] + '<MASK>' + row[1][match.end():]
            data.append((original, edited, masked, float(row[4])))

    train_text, test_text = train_test_split(data, test_size=0.2)

    train_data = pd.DataFrame(
        train_text, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    test_data = pd.DataFrame(
        test_text, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    return train_data, test_data


def tokenize(sentence):
    tokens = tokenizer(sentence, return_tensors='pt',
                       padding='max_length', max_length=max_length)
    for key in tokens:
        tokens[key] = tokens[key].to(device)
    return tokens


def embed_line(tokens):
    return model(**tokens)


def embed_batch(data):
    sentences = list(data)
    embeddings = []
    for idx, sentence in enumerate(sentences):
        inputs = tokenize(sentence)
        # Getting [CLS] token
        embeddings.append(embed_line(inputs)[0][:, 0, :])
    return torch.stack(embeddings).squeeze().to(device)
