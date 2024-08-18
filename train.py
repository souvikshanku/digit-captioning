from dataset import DoubleMnist
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DigitCaptioner


def train(model, train_dataloader, num_epochs=1):
    optimizer = optim.Adam([*model.parameters()], lr=0.003)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        print(f" ---------------- Epoch: {epoch + 1} ----------------")

        for train_features, train_labels in train_dataloader:
            imgs = train_features.clone().detach().float()
            x = train_labels.transpose(0, 1).float()

            out = model.forward(x[:-1], imgs)
            loss = - torch.sum(out * x[1:]) / out.shape[1]
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"checkpoints/epoch_{epoch + 1}.pt")
        print("Checkpint saved...")


if __name__ == "__main__":
    annotations_file = "data/labels.csv"
    img_dir = "data/double_mnist"
    batch_size = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = DoubleMnist(annotations_file, img_dir, device)
    train_dataloader = DataLoader(data, batch_size, shuffle=True)
    model = DigitCaptioner(input_size=29, hidden_size=256, device=device)
    train(model, train_dataloader, num_epochs=1)
