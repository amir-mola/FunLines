import torch
import pdb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import csv
import re
import tqdm
import multiprocessing
import pickle


batch_size = 32
device = torch.device('cuda')

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaModel.from_pretrained('roberta-base').to(device)

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
            original_word = row[1][match.start()+1:match.end()-2]
            original = ' '.join((row[1][:match.start()] + original_word + row[1][match.end():]).split())
            edited = ' '.join((row[1][:match.start()] + row[2] + row[1][match.end():]).split())
            masked = ' '.join((row[1][:match.start()] + '[MASK]' + row[1][match.end():]).split())
            data.append(((original, original_word), (edited, row[2]), (masked, '[MASK]'), float(row[4])))
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    val, test = train_test_split(test, test_size=0.5, random_state=1)
    train = pd.DataFrame(train, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    val = pd.DataFrame(val, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    test = pd.DataFrame(test, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])

    return train, val, test

def tokenize(sentence, tokenizer):
    tokens = tokenizer(sentence, return_tensors='pt')
    for key in tokens:
        tokens[key] = tokens[key].to(device)
    return tokens


def embed_line(tokens, model):
    return model(**tokens)


def embed_batch(original, edited, masked, tokenizer, model):
    original, original_words, edited, edited_words, masked, masked_words = list(original[0]), list(original[1]), \
                                                list(edited[0]), list(edited[1]), list(masked[0]), list(masked[1])
    original_embeddings, edited_embeddings, masked_embeddings = [], [], []
    if tokenizer.name_or_path != 'bert-base-uncased':
        for idx, _ in enumerate(masked_words):
            masked_words[idx] = '<mask>'

        for i, _ in enumerate(masked):
            masked[i] = masked[i].replace('[MASK]', '<mask>')


    for idx, _ in enumerate(original):

        # First find out where masked word is located in the sentence
        inputs_ow = tokenize(original_words[idx], tokenizer)
        inputs_ew = tokenize(edited_words[idx], tokenizer)
        inputs_mw = tokenize(masked_words[idx], tokenizer)

        # Tokenizing original, edited, masked sentences
        inputs_o = tokenize(original[idx], tokenizer)
        inputs_e = tokenize(edited[idx], tokenizer)
        inputs_m = tokenize(masked[idx], tokenizer)

        assert len(inputs_mw['input_ids'][0]) == 3, "Masked word token should only have 3"
        start_index = inputs_m['input_ids'][0]==inputs_mw['input_ids'][0][1]

        start_index = start_index.nonzero().item()
        assert start_index != -1

        end_index_o = start_index + len(inputs_ow['input_ids'][0]) - 2
        end_index_e = start_index + len(inputs_ew['input_ids'][0]) - 2
        # Make sure everything is same until start index

        assert torch.equal(inputs_o['input_ids'][0][:start_index],inputs_e['input_ids'][0][:start_index]) == True
        assert torch.equal(inputs_o['input_ids'][0][:start_index],inputs_m['input_ids'][0][:start_index]) == True
    
        # Getting index of 
        original_em = embed_line(inputs_o, model)[0]
        edited_em = embed_line(inputs_e, model)[0]
        masked_em = embed_line(inputs_m, model)[0]

        assert len(original_em[0]) == len(inputs_o['input_ids'][0])
        assert len(edited_em[0]) == len(inputs_e['input_ids'][0])
        assert len(masked_em[0]) == len(inputs_m['input_ids'][0])

        original_em = original_em[:,start_index:end_index_o,:]
        edited_em = edited_em[:,start_index:end_index_e,:]
        masked_em = masked_em[:,start_index,:]
        original_embeddings.append(original_em)
        edited_embeddings.append(edited_em)
        masked_embeddings.append(masked_em)
    return original_embeddings, edited_embeddings, masked_embeddings


def save_embedding(edited, original, masked, labels, file_name):
    with open('../data/'+file_name, 'a+b') as f:
        pickle.dump((edited, original, masked, labels), f)


train, val, test = create_dataset('../data/train_lines.csv')
trainset = FunLineDataset(train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

valset = FunLineDataset(val)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=batch_size, shuffle=True)

testset = FunLineDataset(test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)

for batch in tqdm.tqdm(trainloader, desc=f'Bert Embedding for trainset '):
    original, edited, masked, labels = batch
    original_embed, edited_embed, masked_embed = embed_batch(original, edited, masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'trainset_bert_embeddings2')

for batch in tqdm.tqdm(valloader, desc=f'Bert Embedding for valset '):
    original, edited, masked, labels = batch
    original_embed, edited_embed, masked_embed = embed_batch(original, edited, masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'valset_bert_embeddings2')

for batch in tqdm.tqdm(testloader, desc=f'Bert Embedding for testset '):
    original, edited, masked, labels = batch
    original_embed, edited_embed, masked_embed = embed_batch(original, edited, masked, tokenizer_bert, model_bert)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'testset_bert_embeddings2')

for batch in tqdm.tqdm(trainloader, desc=f'Roberta Embedding for trainset '):
    original, edited, masked, labels = batch
    original_embed, edited_embed, masked_embed = embed_batch(original, edited, masked, tokenizer_roberta, model_roberta)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'trainset_robert_embeddings2')

for batch in tqdm.tqdm(valloader, desc=f'Roberta Embedding for valset '):
    original, edited, masked, labels = batch
    original_embed, edited_embed, masked_embed = embed_batch(original, edited, masked, tokenizer_roberta, model_roberta)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'valset_robert_embeddings2')

for batch in tqdm.tqdm(testloader, desc=f'Roberta Embedding for testset '):
    original, edited, masked, labels = batch
    original_embed, edited_embed, masked_embed = embed_batch(original, edited, masked, tokenizer_roberta, model_roberta)
    save_embedding(edited_embed, original_embed, masked_embed, labels, 'testset_robert_embeddings2')
