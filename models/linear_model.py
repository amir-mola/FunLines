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

hidden_size = 512
batch_size = 64
num_epochs = 5
learning_rate = 0.001
WEIGHT_DECAY = 0.0005
device = torch.device('cuda')
# TODO this is a basic linear regression model, we can probably get
# higher accuracy with LSTM


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearModel():
    def __init__(self, train_data, test_data):
        self.net = Net().to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    def train(self):
        i = 0
        for epoch in range(num_epochs):
            train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            for idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f'Training {epoch+1} epoch')):
                x, y = self.get_pair(batch)
                y = y.to(device)
                x = embedding.embed_batch(x)
                y_hat = self.net(x)
                loss = self.loss_fn(y_hat, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if idx % (4000 / batch_size) == 0:
                    print(loss.item())
            self.test()
        print('done training')

    def test(self):
        self.net.eval()

        tolerance = 0.2
        correct = 0
        total = 0
        losses = []
        MAE = []
        # In case of python < 3.5
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        with torch.no_grad():
            test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
            for idx, batch in enumerate(tqdm.tqdm(test_loader, desc=f'Test ')):
                x, y = self.get_pair(batch)
                x = embedding.embed_batch(x)
                y_hat = self.net(x)
                y = y.to(device)
                losses.append(self.loss_fn(y_hat, y))
                MAE.append(F.l1_loss(y_hat, y))
                # for b in range(batch_size):
                #     yb = y[b].item()
                #     yhb = y_hat[b].item()
                #     if isclose(yb, yhb, rel_tol=tolerance):
                #         correct += 1
                #     total += 1

        # print(f'Test accuracy: {correct / total}')
        print(f'Average MSE loss for Test: {torch.mean(torch.Tensor(losses))}')
        print(f"Average MAE loss for Test: {torch.mean(torch.Tensor(MAE))}")
    def score(self, seq):
        tokens = embedding.tokenize(seq)
        embedded_seq = embedding.embed_line(tokens)
        x = embedded_seq[0][:, 0, :]
        y_hat = self.net(x)
        return y_hat.item()

    @staticmethod
    def get_pair(batch):
        x, y = batch
        y = torch.Tensor(y.float())
        y = torch.unsqueeze(y, 1)
        return x, y

    @staticmethod
    def load_data():
        train, test = embedding.create_dataset('data/train_funlines.csv')
        print('done splitting')
        data_train = embedding.FunLineDataset(train)
        data_test = embedding.FunLineDataset(test)

        # train = embedding.embed(train)
        # test = embedding.embed(test)
        # train = embedding.batch_embedded(train, batch_size)
        # test = embedding.batch_embedded(test, batch_size)
        print('done embedding')
        pickle.dump(data_train, open(f'{cache_dir}/train_embedded.p', 'wb'))
        pickle.dump(data_test, open(f'{cache_dir}/test_embedded.p', 'wb'))
