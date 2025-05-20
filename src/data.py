from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, img_size=(128, 128), num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # MNIST 是單通道，請用以下參數
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])

    train_ds = MNIST('./data', train=True, download=True, transform=transform)
    test_ds  = MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
