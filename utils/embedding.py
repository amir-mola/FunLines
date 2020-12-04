import torch
import pdb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import csv
import re
import tqdm
import multiprocessing

device = torch.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased', return_dict=True).to(device)

max_length = 42

class FunLineDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(FunLineDataset, self).__init__()
        self.data = data

    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data['Text'][idx], self.data['Mean_Score'][idx]





# Creates train and test data in pands dataframe
def create_dataset(path):

    data = []
    with open(path, encoding="utf8", errors='ignore') as f:
        rows = csv.reader(f)
        rows = list(rows)
        for row in rows[1:]:
            if float(row[4]) == 0: continue
            match = re.search(r'<.*>', row[1])
            sentence = row[1][:match.start()] + row[2] + row[1][match.end():]
            data.append((sentence, float(row[4])))

    train_text, test_text = train_test_split(data, test_size=0.2)

    train_data = pd.DataFrame(train_text, columns=['Text', 'Mean_Score'])
    test_data = pd.DataFrame(test_text, columns=['Text', 'Mean_Score'])
    return train_data, test_data


# Embeds sentences through language model
# def embed(data_set):
#     sentences = list(data_set['Text'])
#     scores = list(data_set['Mean_Score'])
#     embeddings = []

#     for idx, sentence in enumerate(tqdm.tqdm(sentences, desc="Embedding: ")):
#         inputs = tokenize(sentence)
#         embeddings.append((embed_line(inputs), scores[idx]))
#     return embeddings


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
        embeddings.append(embed_line(inputs)[0][:,0,:])
    return torch.stack(embeddings).squeeze().to(device)

# Returns batch of (tensor of batch_size x embed size, batch_size x mean_score)
# def batch_embedded(data, size):
#     random.shuffle(data)
#     num_batch = len(data)//size

#     batches = []
#     for i in range(num_batch):
#         batch_features = []
#         batch_scores = []

#         for (states, scores) in data[i*size:(i+1)*size]:
#             # Extract [CLS] embedding
#             batch_features.append(states[0][:, 0, :])
#             batch_scores.append(scores)

#         batches.append((
#             torch.stack(batch_features).squeeze(),
#             batch_scores
#         ))

#     # Putting last batch which is smaller than others
#     if len(data[(i+1)*size:]) > 0:
#         batch_features = []
#         batch_scores = []

#         for (states, scores) in data[(i+1)*size:]:
#             # Extract [CLS] embedding
#             batch_features.append(states[0][:, 0, :])
#             batch_scores.append(scores)

#         batches.append((
#             torch.stack(batch_features).squeeze(),
#             batch_scores
#         ))

#     return batches

