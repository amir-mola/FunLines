import torch
import pdb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
import csv
import re
import tqdm
import multiprocessing
import pickle


batch_size = 32
device = torch.device('cuda')



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
        for row in rows[1:]:
            if float(row[4]) == 0: continue
            match = re.search(r'<.*>', row[1])
            original = row[1][:match.start()] + row[1][match.start()+1:match.end()-2] + row[1][match.end():]
            edited = row[1][:match.start()] + row[2] + row[1][match.end():]
            masked = row[1][:match.start()] + '<MASK>' + row[1][match.end():]
            data.append((original, edited, masked, float(row[4])))
    train, test = train_test_split(data)
    train = pd.DataFrame(train, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    test = pd.DataFrame(test, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    return train, test

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
    for sentence in sentences:
        inputs = tokenize(sentence)
        # Getting [CLS] token
        embedded = embed_line(inputs)
        embedded = (embedded[0], embedded[1])
        embeddings.append(embedded)
    return embeddings


def save_embedding(original, edited, masked, labels, file_name):
    with open('../data/'+file_name, 'a+b') as f:
        pickle.dump((original, edited, masked, labels), f)

train, test = create_dataset('../data/train_lines.csv')

trainset = FunLineDataset(train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = FunLineDataset(test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

for batch in tqdm.tqdm(trainloader, desc=f'Embedding for trainset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original)
    edited_embed = embed_batch(edited)
    masked_embed = embed_batch(masked)
    save_embedding(original_embed, edited_embed, masked_embed, labels, 'trainset_embeddings')

for batch in tqdm.tqdm(testloader, desc=f'Embedding for testset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original)
    edited_embed = embed_batch(edited)
    masked_embed = embed_batch(masked)
    save_embedding(original_embed, edited_embed, masked_embed, labels, 'testset_embeddings')
