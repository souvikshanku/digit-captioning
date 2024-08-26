import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, device, emb_size):
        super(CNN, self).__init__()
        self.device = device
        self.emb_size = emb_size

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.img_fc = nn.Linear(16 * 15 * 15, self.emb_size)

    def forward(self, img):
        img = F.relu(self.conv1(img))
        img = self.pool(img)
        img = F.relu(self.conv2(img))
        img = self.pool(img)
        img = torch.flatten(img, 1)
        img = F.relu(self.img_fc(img))
        return img


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.xh = nn.Linear(input_size, hidden_size * 4)
        self.hh = nn.Linear(hidden_size, hidden_size * 4)
        self.out_fc = nn.Linear(hidden_size, input_size)

    def forward(self, inp, img):
        # inp shape: seq length, batch size, vocab size
        output = []
        img = torch.cat((img, img, img, img), 1)

        seq_length = inp.shape[0]
        batch_size = inp.shape[1]

        c_t_minus_1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        h_t_minus_1 = torch.zeros(batch_size, self.hidden_size).to(self.device)

        for t in range(seq_length):
            x_t = inp[t]

            gates = self.xh(x_t) + self.hh(h_t_minus_1) + img * (t == 0)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

            i_t = torch.sigmoid(input_gate)
            f_t = torch.sigmoid(forget_gate)
            o_t = torch.sigmoid(output_gate)
            g_t = torch.tanh(cell_gate)

            c_t = (f_t * c_t_minus_1) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            c_t_minus_1 = c_t
            h_t_minus_1 = h_t

            out = self.out_fc(h_t)
            output.append(out)

        output = torch.stack(output)

        return F.log_softmax(output, dim=-1)


class DigitCaptioner(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(DigitCaptioner, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cnn = CNN(self.device, self.hidden_size)
        self.lstm = LSTM(self.input_size, self.hidden_size, self.device)

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inp, img):
        # inp shape: seq length, batch size, vocab size
        img = self.cnn.forward(img)
        out = self.lstm.forward(inp, img)

        return out
