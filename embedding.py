import torch
import pdb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import csv
import re


device = torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)


# Creates train and test data in pands dataframe
def create_dataset(path):

    data = []
    with open(path, encoding="utf8", errors='ignore') as f:
        rows = csv.reader(f)
        rows = list(rows)
        for row in rows[1:100]:
            match = re.search(r'<.*>', row[1])
            sentence = row[1][:match.start()] + row[2] + row[1][match.end():]
            data.append((sentence, float(row[4])))

    train_text, test_text = train_test_split(data, test_size=0.2)

    train_data = pd.DataFrame(train_text, columns=['Text', 'Mean_Score'])
    test_data = pd.DataFrame(test_text, columns=['Text', 'Mean_Score'])
    return train_data, test_data

# Embeds sentences through language model
def embed(data_set):
    sentences = list(data_set['Text'])
    scores = list(data_set['Mean_Score'])
    embeddings = []

    for idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors='pt', padding='max_length', max_length = 42)
        for key in inputs: inputs[key] = inputs[key].to(device)
        embeddings.append((model(**inputs), scores[idx]))

    return embeddings

# Creates random batches
# This will return batch (tensor of sequences(batch_size x max_length x embedd_size), 
# tensor of representation (batch_size x embedd_size), mean_score)
def create_batch(data, size):
    random.shuffle(data)
    num_batch = len(data)//size
    
    batches = []
    for i in range(num_batch):
        batch_sequence = []
        batch_representation = []
        batch_scores = []
        for item in data[i*size:(i+1)*size]:
            batch_sequence.append(item[0][0])
            batch_representation.append(item[0][1])
            batch_scores.append(item[1])
        batches.append((torch.stack(batch_sequence).squeeze(), torch.stack(batch_representation).squeeze(), batch_scores))

    # Putting last batch which is smaller than others
    if len(data[(i+1)*size:]) > 0:
        batch_sequence = []
        batch_representation = []
        batch_scores = []
        for item in data[(i+1)*size:]:
            batch_sequence.append(item[0][0])
            batch_representation.append(item[0][1])
            batch_scores.append(item[1])
        batches.append((torch.stack(batch_sequence).squeeze(), torch.stack(batch_representation).squeeze(), batch_scores))
    return batches

batch_size = 4

train, test = create_dataset('train_funlines.csv')
embeded = embed(train)
batches = create_batch(embeded, batch_size)
pdb.set_trace()

