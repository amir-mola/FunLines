import torch
import pdb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, XLNetTokenizer, XLNetModel
import csv
import re
import tqdm
import multiprocessing
import pickle


batch_size = 32


<<<<<<< HEAD
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaModel.from_pretrained('roberta-base').to(device)
tokenizer_xlnet = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model_xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
=======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>>>>>>> 64d79087ea58a13effcb213d3636ecf30aa69194


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
            if float(row[4]) == 0:
                continue
            match = re.search(r'<.*>', row[1])
            original = row[1][:match.start()] + row[1][match.start() +
                                                       1:match.end()-2] + row[1][match.end():]
            edited = row[1][:match.start()] + row[2] + row[1][match.end():]
            masked = row[1][:match.start()] + '<MASK>' + row[1][match.end():]
            data.append((original, edited, masked, float(row[4])))
    train, test = train_test_split(data)
    train = pd.DataFrame(
        train, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    test = pd.DataFrame(
        test, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    return train, test

<<<<<<< HEAD
def tokenize(sentence, tokenizer):
=======

def tokenize(sentence):
>>>>>>> 64d79087ea58a13effcb213d3636ecf30aa69194
    tokens = tokenizer(sentence, return_tensors='pt',
                       padding='max_length', max_length=max_length)
    for key in tokens:
        tokens[key] = tokens[key].to(device)
    return tokens


def embed_line(tokens, model):
    return model(**tokens)


def embed_batch(data, tokenizer, model):
    sentences = list(data)
    embeddings = []
    for sentence in sentences:
        inputs = tokenize(sentence, tokenizer)
        # Getting [CLS] token
        embedded = embed_line(inputs, model)
        embedded = (embedded[0], embedded[1])
        embeddings.append(embedded)
    return embeddings


def save_embedding(edited, original, masked, labels, file_name):
    with open('../data/'+file_name, 'a+b') as f:
        pickle.dump((edited, original, masked, labels), f)


train, test = create_dataset('../data/train_lines.csv')

trainset = FunLineDataset(train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

testset = FunLineDataset(test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)

for batch in tqdm.tqdm(trainloader, desc=f'Bert Embedding for trainset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original, tokenizer_bert, model_bert)
    edited_embed = embed_batch(edited, tokenizer_bert, model_bert)
    masked_embed = embed_batch(masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'trainset_bert_embeddings')

for batch in tqdm.tqdm(testloader, desc=f'Bert Embedding for testset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original, tokenizer_bert, model_bert)
    edited_embed = embed_batch(edited, tokenizer_bert, model_bert)
    masked_embed = embed_batch(masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'testset_bert_embeddings')

for batch in tqdm.tqdm(trainloader, desc=f'Roberta Embedding for trainset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original, tokenizer_bert, model_bert)
    edited_embed = embed_batch(edited, tokenizer_bert, model_bert)
    masked_embed = embed_batch(masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'trainset_robert_embeddings')

for batch in tqdm.tqdm(testloader, desc=f'Roberta Embedding for testset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original, tokenizer_bert, model_bert)
    edited_embed = embed_batch(edited, tokenizer_bert, model_bert)
    masked_embed = embed_batch(masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'testset_robert_embeddings')

for batch in tqdm.tqdm(trainloader, desc=f'Xlnet Embedding for trainset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original, tokenizer_bert, model_bert)
    edited_embed = embed_batch(edited, tokenizer_bert, model_bert)
    masked_embed = embed_batch(masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'trainset_xlnet_embeddings')

for batch in tqdm.tqdm(testloader, desc=f'Xlnet Embedding for testset '):
    original, edited, masked, labels = batch
    original_embed = embed_batch(original, tokenizer_bert, model_bert)
    edited_embed = embed_batch(edited, tokenizer_bert, model_bert)
    masked_embed = embed_batch(masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'testset_xlnet_embeddings')
