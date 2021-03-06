import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #print("input size {}, hidden size {}, num_layers {}".format(input_size, hidden_size, num_layers))
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        #print("shape")
        #print(x.shape)
        #assert False
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class Model(object):
    def __init__(self, debug=False):
        return

    def train(self, data, labels, hyperparams=dict()):
        '''
        data : list of input
        labels : list of corresponding output
        hyperparams = dictionary maps from name to its value
        return : None
        '''

        train_data, train_labels = data, labels
        val_data, val_labels = hyperparams["validation"]
        epochs = hyperparams["epochs"]
        batch_size = hyperparams["batch_size"]

        train_data = np.transpose(train_data, (0, 2, 1))
        val_data = np.transpose(val_data, (0, 2, 1))

        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
        tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=4, pin_memory=True)

        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(val_data), torch.LongTensor(val_labels))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=4, pin_memory=True)

        self.model = RNN(6, 6, 2, 2).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        best_acc = 0
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print('=== Epoch: {} ==='.format(epoch))
            train_loss = []
            train_acc = []
            val_loss = []
            val_acc = []
            tr_iter = iter(tr_dataloader)
            self.model.train()
            correct = 0
            total = 0
            for batch in tqdm(tr_iter):
                optimizer.zero_grad()
                x, y = batch
                x, y = x.to(device), y.to(device)
                model_output = self.model(x)
                loss = criterion(model_output, y)
                _, predicted = torch.max(model_output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            avg_loss = np.mean(train_loss)
            avg_acc = float(correct) / total
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
            if val_dataloader is None:
                continue
            val_iter = iter(val_dataloader)
            self.model.eval()
            correct = 0
            total = 0
            for batch in val_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)
                model_output = self.model(x)
                loss = criterion(model_output, y)
                _, predicted = torch.max(model_output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum()
                val_loss.append(loss.item())
            avg_loss = np.mean(val_loss)
            avg_acc = float(correct) / total
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                best_acc = avg_acc
                best_state = self.model.state_dict()

        return

    def test(self, data, labels):
        '''
        data : list of input

        return : list of predicted output
        '''
        batch_size = 800
        data = np.transpose(data, (0, 2, 1))
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                     shuffle=True, num_workers=4, pin_memory=True)

        correct = 0
        total = 0
        avg_conf = []
        for epoch in range(1):
            test_iter = iter(test_dataloader)
            for batch in test_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)
                model_output = self.model(x)
                _, predicted = torch.max(model_output.data, 1)
                conf_mat = confusion_matrix(y.data.cpu().numpy(), predicted.data.cpu().numpy())
                print(conf_mat)
                total += y.size(0)
                correct += (predicted == y).sum()
                avg_acc = float(correct) / total
                print('Test Acc: {}'.format(avg_acc))
                return conf_mat

        #return confusion_matrix(labels, labels_pred)
