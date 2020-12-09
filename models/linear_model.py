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
device = torch.device('cuda')
# TODO this is a basic linear regression model, we can probably get
# higher accuracy with LSTM


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
                original, edited, masked, y = self.get_pair(batch)
                y = y.to(device)
                original = embedding.embed_batch(original)
                edited = embedding.embed_batch(edited)
                masked = embedding.embed_batch(masked)
                x = torch.cat((original, edited, masked),1)
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
        RMSE = []
        # In case of python < 3.5
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        with torch.no_grad():
            test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
            for idx, batch in enumerate(tqdm.tqdm(test_loader, desc=f'Test ')):
                original, edited, masked, y = self.get_pair(batch)
                original = embedding.embed_batch(original)
                edited = embedding.embed_batch(edited)
                masked = embedding.embed_batch(masked)
                x = torch.cat((original, edited, masked),1)
                y_hat = self.net(x)
                y = y.to(device)
                losses.append(self.loss_fn(y_hat, y))
                RMSE.append(torch.sqrt(self.loss_fn(y_hat, y)))
                # for b in range(batch_size):
                #     yb = y[b].item()
                #     yhb = y_hat[b].item()
                #     if isclose(yb, yhb, rel_tol=tolerance):
                #         correct += 1
                #     total += 1

        # print(f'Test accuracy: {correct / total}')
        print(f'Average MSE loss for Test: {torch.mean(torch.Tensor(losses))}')
        print(f"Average MAE loss for Test: {torch.mean(torch.Tensor(RMSE))}")
    def score(self, seq):
        tokens = embedding.tokenize(seq)
        embedded_seq = embedding.embed_line(tokens)
        x = embedded_seq[0][:, 0, :]
        y_hat = self.net(x)
        return y_hat.item()

    @staticmethod
    def get_pair(batch):
        original, edited, masked, y = batch
        y = torch.Tensor(y.float())
        y = torch.unsqueeze(y, 1)
        return original, edited, masked, y

    @staticmethod
    def load_data():
        train, test = embedding.create_dataset('data/train_lines.csv')
        print('done splitting')
        data_train = embedding.FunLineDataset(train)
        data_test = embedding.FunLineDataset(test)

        print('done embedding')
        pickle.dump(data_train, open(f'{cache_dir}/train_embedded.p', 'wb'))
        pickle.dump(data_test, open(f'{cache_dir}/test_embedded.p', 'wb'))
