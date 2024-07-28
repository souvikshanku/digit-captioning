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
        self.num_layers = 1

        self.W_ih = nn.Parameter(torch.randn(
            (self.num_layers, self.hidden_size, self.vocab_size)
        ))
        self.b_ih = nn.Parameter(torch.zeros(
            (self.num_layers, self.hidden_size)
        ))
        self.W_hh = nn.Parameter(torch.randn(
            (self.num_layers, self.hidden_size, self.hidden_size)
        ))
        self.b_hh = nn.Parameter(torch.zeros(
            (self.num_layers, self.hidden_size)
        ))

        self.W_ho = nn.Parameter(torch.randn(
            (self.vocab_size, self.hidden_size)
        ))
        self.b_ho = nn.Parameter(torch.zeros(
            (self.vocab_size)
        ))

    def forward(self, x, img_emb):
        batch_size = x.shape[1]
        output = []
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0.clone()
        h_t = h_0.clone()

        for t in range(self.seq_len):
            for layer in range(self.num_layers):
                h_t[layer] = torch.tanh(
                    x[t] @ self.W_ih[layer].T
                    + self.b_ih[layer]
                    + h_t_minus_1[layer] @ self.W_hh[layer].T
                    + self.b_hh[layer]
                    + (t == 0) * img_emb
                )

            output.append(h_t[-1].clone())
            h_t_minus_1 = h_t.clone()

        output = torch.stack(output)
        output = output.view((self.seq_len, batch_size, self.hidden_size))
        output = output @ self.W_ho.T + self.b_ho

        return F.log_softmax(output, dim=-1)
