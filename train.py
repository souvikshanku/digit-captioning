from dataset import DoubleMnist
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import models
# from models import CNN, LSTM
from model import CaptionModel


def train(train_dataloader, num_epochs=3):

    model = CaptionModel(input_size=29, hidden_size=256)
    ckpt = 0

    ckpt = 24
    checkpoint = torch.load(f"checkpoints/itr_{ckpt}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(
        [*model.parameters()],
        lr=0.0001
    )

    for epoch in range(num_epochs):
        print(f" ---------------- Epoch: {ckpt + epoch + 1} ----------------")

        for train_features, train_labels in train_dataloader:

            imgs = train_features.clone().detach().float()

            x = train_labels.transpose(0, 1).float()

            # print(img_emb.shape)
            # print(x.shape)
            # print("--------------------------")

            out = model.forward(x[:-1], imgs)
            # print(out.shape)

            loss = - torch.sum(out * x[1:]) / out.shape[1]
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=5)
            # torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=5)
            # for param in list(cnn.parameters()):
            #     print(param.grad.sum())

            # if ckpt + epoch >= 3:
            #     print("", model.conv1.weight.grad.sum(), "------------")
            #     print("", model.img_fc.weight.grad.sum(), "------------\n\n")

            optimizer.step()

        torch.save({
            "iter": ckpt + epoch + 1,
            "model_state_dict": model.state_dict(),
        }, f"checkpoints/itr_{ckpt + epoch + 1}.pt")
        print("Checkpint saved...")


if __name__ == "__main__":
    annotations_file = "data/labels.csv"
    img_dir = "data/double_mnist"
    batch_size = 1024

    data = DoubleMnist(annotations_file, img_dir)
    train_set, test_set = torch.utils.data.random_split(data, [80000, 20000])

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True)

    train(train_dataloader, num_epochs=20)
