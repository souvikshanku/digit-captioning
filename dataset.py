import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class DoubleMnist(Dataset):
    def __init__(self, annotations_file, img_dir, device):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.device = device

        vocab = "abcdefghijklmnopqrstuvwxyz "
        self.vocab = dict(zip(list(vocab), range(len(vocab))))
        self.vocab["<start>"] = 27
        self.vocab["<end>"] = 28

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        label = self.img_labels.iloc[idx, 2]
        label = self._get_ohe_label(label)

        image = image.to(self.device)
        label = label.to(self.device)

        return image, label

    def _get_ohe_label(self, label):
        tokens = [27]  # <start>
        tokens += [self.vocab[char] for char in label]
        extra = 15 - len(tokens)
        tokens += [28] * extra  # <end>

        num_classes = len(self.vocab)
        y = F.one_hot(torch.tensor(tokens), num_classes)

        return y
