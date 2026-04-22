from torch.utils.data import DataLoader
from torchvision import datasets, transforms


NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
BATCH_SIZE = 64
LEARNING_RATE = 0.001

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD)
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform_train,
    download=True
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform_test,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
)

classes = train_dataset.classes
