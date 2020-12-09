'''
This is to save embedding for later use.
Still in progress
'''




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
import pickle


batch_size = 32
device = torch.device('cuda')



tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

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

    data = pd.DataFrame(data, columns=['Original', 'Edited', 'Masked', 'Mean_Score'])
    return data

def tokenize(sentence):
    tokens = tokenizer(sentence, return_tensors='pt',
                       padding='max_length', max_length=max_length)
    for key in tokens:
        tokens[key] = tokens[key]
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


dataset = create_dataset('../data/train_lines.csv')

dataset = FunLineDataset(dataset)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
original_embed = []
edited_embed = []
masked_embed = []

for batch in tqdm.tqdm(loader, desc=f'Embedding '):
    original, edited, masked, _ = batch
    original_embed.extend(embed_batch(original))
    edited_embed.extend(embed_batch(edited))
    masked_embed.extend(embed_batch(masked))


with open('original_embedding', 'wb') as f:
    pickle.dump(original_embed, f)

with open('edited_embedding', 'wb') as f:
    pickle.dump(edited_embed, f)

with open('masked_embedding', 'wb') as f:
    pickle.dump(masked_embed, f)

with open('original_embedding', 'rb') as f:
    test = pickle.load(f)
