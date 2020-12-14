import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from utils import embedding
from utils.cachedir import cache_dir

import numpy as np
import pickle
import pdb
import tqdm

input_size = 768
output_size = 1

hidden_size = 128
num_epochs = 10
learning_rate = 0.0005
WEIGHT_DECAY = 0.0005
device = torch.device('cuda')

ninp = 768
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 6 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value


# TODO this is a Transformer Model

class Transformer(nn.Module):

    def __init__(self, ninp, nhead, nlayers, dropout=0.5):
        super(Transformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(ninp, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(1, input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, src):
        # Creating cls_token to get representation of all sentence embeddings
        cls_token = torch.zeros(len(src), 1,dtype=torch.long).to(device)
        cls_token = self.embedding(cls_token)
        x = torch.cat((cls_token,src), dim=1)
        x = self.transformer_encoder(x)
        # Just taking first embedidng. Hopefully, it captures other data by going through transformer layers
        x = x[:,0,:]
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerModel():
    def __init__(self):
        self.net = Transformer(ninp, nhead, nlayers).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    def train(self):

        for epoch in range(num_epochs):
            print("Training : ", epoch+1)
            bertfile = open('data/trainset_bert_embeddings', 'rb')
            robertfile = open('data/trainset_robert_embeddings', 'rb')
            xlnetfile = open('data/trainset_xlnet_embeddings', 'rb')
            while True:
                try:
                    x_bert, y = self.get_pair(pickle.load(bertfile))
                    x_roberta, _ = self.get_pair(pickle.load(robertfile))
                    x_xlnet, _ = self.get_pair(pickle.load(xlnetfile))
                    y = y.to(device)
                    x = torch.cat((x_roberta, x_bert, x_xlnet), dim=1)

                    y_hat = self.net(x)
                    loss = torch.sqrt(self.loss_fn(y_hat, y))

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                except (EOFError, pickle.UnpicklingError):
                    break
            bertfile.close()
            robertfile.close()
            xlnetfile.close()
            self.test()

        print('done training')

    def test(self):
        self.net.eval()
        print("Test")
        losses = []

        with torch.no_grad():
            bertfile = open('data/testset_bert_embeddings', 'rb')
            robertfile = open('data/testset_robert_embeddings', 'rb')
            xlnetfile = open('data/testset_xlnet_embeddings', 'rb')
            while True:
                try:
                    x_bert, y = self.get_pair(pickle.load(bertfile))
                    x_roberta, _ = self.get_pair(pickle.load(robertfile))
                    x_xlnet, _ = self.get_pair(pickle.load(xlnetfile))
                    y = y.to(device)
                    x = torch.cat((x_roberta, x_bert, x_xlnet), dim=1)
                    y_hat = self.net(x)
                    loss = self.loss_fn(y_hat, y)
                    losses.append(loss)

                except (EOFError, pickle.UnpicklingError):
                    break
            bertfile.close()
            robertfile.close()
            xlnetfile.close()
        print(f'Average MSE loss for Test: {torch.mean(torch.Tensor(losses))}')
        print(f"Average RMSE loss for Test: {torch.sqrt(torch.mean(torch.Tensor(losses)))}")


    @staticmethod
    def get_pair(batch):
        original, edited, masked, y = batch
        stacked_batch = []

        for i in range(len(original)):
            stacked_batch.append(torch.stack([original[i][0][:,0,:],
            edited[i][0][:,0,:], masked[i][0][:,0,:]]).squeeze())

        x = torch.stack(stacked_batch)
        y = y.to(torch.float)
        y = torch.unsqueeze(y, 1)
        return x, y

