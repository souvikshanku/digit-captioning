from dataset import DoubleMnist
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import models
from models import RNN, CNN


def train(train_dataloader, num_epochs=3):
    # transfer learning
    # cnn = models.vgg16(weights="DEFAULT")
    # for param in cnn.parameters():
    #     param.requires_grad = False
    # cnn.classifier[6] = torch.nn.Linear(4096, 50)

    cnn = CNN()
    rnn = RNN(seq_len=15 - 1)
    ckpt = 0

    # ckpt = 2
    # checkpoint = torch.load(f"checkpoints/itr_{ckpt}.pt")
    # cnn.load_state_dict(checkpoint['cnn_state_dict'])
    # rnn.load_state_dict(checkpoint['rnn_state_dict'])

    optimizer = optim.Adam([*cnn.parameters(), *rnn.parameters()], lr=0.0005)

    for epoch in range(num_epochs):
        print(f" ---------------- Epoch: {ckpt + epoch + 1} ----------------")

        for train_features, train_labels in train_dataloader:

            imgs = train_features.clone().detach().float()
            img_emb = cnn.forward(imgs)
            x = train_labels.transpose(0, 1).float()

            # print(img_emb.shape)
            # print(x.shape)
            # print("--------------------------")

            out = rnn.forward(x[:-1], img_emb)
            # print(out.shape)

            loss = - torch.sum(out * x[1:]) / out.shape[1]
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=5)
            optimizer.step()

        torch.save({
            "iter": ckpt + epoch + 1,
            "cnn_state_dict": cnn.state_dict(),
            "rnn_state_dict": rnn.state_dict(),
        }, f"checkpoints/itr_{ckpt + epoch + 1}.pt")
        print("Checkpint saved...")

    return cnn, rnn


if __name__ == "__main__":
    annotations_file = "data/labels.csv"
    img_dir = "data/double_mnist"
    batch_size = 100

    data = DoubleMnist(annotations_file, img_dir)
    train_set, test_set = torch.utils.data.random_split(data, [80000, 20000])

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True)

    train(train_dataloader, num_epochs=20)
