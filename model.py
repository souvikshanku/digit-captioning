import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 50)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # 64 x 64 x 16
        x = self.pool(x)                # 32 x 32 x 16
        x = F.relu(self.conv2(x))       # 28 x 28 x 8
        x = self.pool(x)                # 14 x 14 x 8
        x = torch.flatten(x, 1)         # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
"""


class CaptionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CaptionModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # self.img_fc1 = nn.Linear(1024, 512)  # ######
        # self.img_fc2 = nn.Linear(512, self.hidden_size)

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.img_fc = nn.Linear(16 * 15 * 15, self.hidden_size)  # ######

        self.xh = nn.Linear(input_size, hidden_size * 4)
        self.hh = nn.Linear(hidden_size, hidden_size * 4)

        self.out_fc = nn.Linear(hidden_size, self.input_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inp, img_emb):
        # inp shape: seq length, batch size, vocab size
        output = []

        seq_length = inp.shape[0]
        batch_size = inp.shape[1]

        # img_emb = nn.AvgPool2d(2, 2)(img_emb)

        img_emb = F.relu(self.conv1(img_emb))
        img_emb = self.pool(img_emb)
        img_emb = F.relu(self.conv2(img_emb))
        img_emb = self.pool(img_emb)
        img_emb = torch.flatten(img_emb, 1)
        img_emb = F.relu(self.img_fc(img_emb))

        # img_emb = F.relu(self.img_fc2(img_emb))

        img_emb = torch.cat((img_emb, img_emb, img_emb, img_emb), 1)

        c_t_minus_1 = torch.zeros(batch_size, self.hidden_size)
        h_t_minus_1 = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            x_t = inp[t]

            gates = self.xh(x_t) + self.hh(h_t_minus_1) + img_emb * (t == 0)
            input_gate, forget_gate, gate_gate, output_gate = gates.chunk(4, 1)

            i_t = torch.sigmoid(input_gate)
            f_t = torch.sigmoid(forget_gate)
            o_t = torch.sigmoid(output_gate)
            g_t = torch.tanh(gate_gate)

            c_t = (f_t * c_t_minus_1) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            c_t_minus_1 = c_t
            h_t_minus_1 = h_t

            out = self.out_fc(h_t)
            output.append(out)

        output = torch.stack(output)

        return F.log_softmax(output, dim=-1)
