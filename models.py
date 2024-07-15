import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)  # same h, w
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 5, 5)
        self.fc1 = nn.Linear(5 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class RNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = 20
        self.vocab_size = 29
        self.batch_size = 64
        self.num_layers = 1

        self.W_hx = torch.randn(
            (self.num_layers, self.hidden_size, self.vocab_size),
            requires_grad=True
        )
        self.b_hx = torch.zeros(
            (self.num_layers, self.hidden_size),
            requires_grad=True
        )
        self.W_hh = torch.randn(
            (self.num_layers, self.hidden_size, self.hidden_size),
            requires_grad=True
        )
        self.b_hh = torch.zeros(
            (self.num_layers, self.hidden_size),
            requires_grad=True
        )

        self.W_oh = torch.randn(
            (self.vocab_size, self.hidden_size), requires_grad=True
        )
        self.b_oh = torch.zeros((self.vocab_size), requires_grad=True)

    def forward(self, x, img_emb):
        output = []
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        h_t_minus_1 = h_0
        h_t = h_0

        for t in range(self.seq_len):
            for layer in range(self.num_layers):
                h_t[layer] = torch.tanh(
                    x[t] @ self.W_hx[layer].T
                    + self.b_hx[layer]
                    + h_t_minus_1[layer] @ self.W_hh[layer].T
                    + self.b_hh[layer]
                    + (t == 1) * img_emb
                )

            output.append(h_t[-1])
            h_t_minus_1 = h_t

        output = torch.stack(output)

        output = output.view((self.seq_len, self.batch_size, self.hidden_size))
        output = output @ self.W_oh.T + self.b_oh

        return F.log_softmax(output, dim=-1)
