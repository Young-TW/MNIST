import argparse
import torch
from gpu import detect_gpu
from data import get_dataloaders
from model import SimpleCNN
from train import train_one_epoch, validate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(detect_gpu())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}: "
            f"Train {train_acc:.4f}/{train_loss:.4f} | "
            f"Test {test_acc:.4f}/{test_loss:.4f}"
        )

    torch.save(model.state_dict(), "cnn_cifar10.pth")


if __name__ == "__main__":
    main()
