import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import embedding
from utils.cachedir import cache_dir

import numpy as np
import pickle
import pdb
import tqdm

input_size = 768
output_size = 1

hidden_size = 768
batch_size = 16
num_epochs = 5
learning_rate = 0.001
WEIGHT_DECAY = 0.0005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3*input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearModel():
    def __init__(self):
        self.net = Net().to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.net.parameters(
        ), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    def train(self):

        for epoch in range(num_epochs):
            print("Training : ", epoch+1)
            infile = open('data/trainset_embeddings', 'rb')
            while True:
                try:
                    x, y = self.get_pair(pickle.load(infile))
                    y = y.to(device)
                    y_hat = self.net(x)
                    loss = self.loss_fn(y_hat, y)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                except (EOFError, pickle.UnpicklingError):
                    break
            infile.close()
            self.test()

        print('done training')

    def test(self):
        self.net.eval()
        print("Test")
        losses = []

        with torch.no_grad():
            infile = open('data/testset_embeddings', 'rb')
            while True:
                try:
                    x, y = self.get_pair(pickle.load(infile))
                    y = y.to(device)

                    y_hat = self.net(x)
                    loss = self.loss_fn(y_hat, y)
                    losses.append(loss)

                except (EOFError, pickle.UnpicklingError):
                    break

        print(f'Average MSE loss for Test: {torch.mean(torch.Tensor(losses))}')
        print(
            f"Average RMSE loss for Test: {torch.sqrt(torch.mean(torch.Tensor(losses)))}")

    @staticmethod
    def get_pair(batch):
        original, edited, masked, y = batch
        original_batch = []
        edited_batch = []
        masked_batch = []
        # Just get CLS token
        for i in range(len(original)):
            original_batch.append(original[i][0][:, 0, :])
            edited_batch.append(edited[i][0][:, 0, :])
            masked_batch.append(masked[i][0][:, 0, :])
        # Create dataset such that batch x size of embedding
        original_batch = torch.stack(original_batch).squeeze().to(device)
        edited_batch = torch.stack(edited_batch).squeeze().to(device)
        masked_batch = torch.stack(masked_batch).squeeze().to(device)

        # Concatenate original embed + edited embed + masked embed
        x = torch.cat((original_batch, edited_batch, masked_batch), 1)
        y = y.to(torch.float)
        y = torch.unsqueeze(y, 1)
        return x, y
