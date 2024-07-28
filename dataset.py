import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class DoubleMnist(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir

        vocab = "abcdefghijklmnopqrstuvwxyz "
        self.vocab = dict(zip(list(vocab), range(len(vocab))))
        self.vocab["<start>"] = 26
        self.vocab["<end>"] = 27

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        label = self._get_ohe_label(label)
        return image, label

    def _get_ohe_label(self, label):
        tokens = [27]  # <start>
        tokens += [self.vocab[char] for char in label]
        extra = 15 - len(tokens)
        tokens += [28] * extra  # <end>

        num_classes = len(self.vocab)
        y = F.one_hot(torch.tensor(tokens), num_classes)

        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    annotations_file = "data/labels.csv"
    img_dir = "data/double_mnist"

    data = DoubleMnist(annotations_file, img_dir)
    train_set, test_set = torch.utils.data.random_split(data, [80000, 20000])

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}, shape:{label.shape}")
