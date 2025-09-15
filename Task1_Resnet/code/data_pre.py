import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

class Add_to_Transform(Dataset):
    """
    train_set -> augmentation and normalization
    valid_set/test_set -> normalization
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

def build_transformer():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        # Augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, eval_transform
def get_dataloaders(config):
    train_transform, test_transform = build_transformer()
    # batch_size
    train_batch_size = config['batch_size']
    valid_batch_size = test_batch_size = 256

    full_train_set = tv.datasets.CIFAR10(root='./data', train=True, download=False, transform=None)
    test_set = tv.datasets.CIFAR10(root='./data', train=False, download=False, transform=None)

    # split train into train and valid
    train_size = int((1-config['valid_ratio']) * len(full_train_set))
    valid_size = len(full_train_set) - train_size
    generator = torch.Generator().manual_seed(config['seed'])
    train_set, valid_set = torch.utils.data.random_split(full_train_set, [train_size, valid_size], generator=generator)

    #! Attention: valid_test should not share the same transform as train_set
    train_set = Add_to_Transform(train_set, train_transform)
    valid_set = Add_to_Transform(valid_set, test_transform)
    test_set = Add_to_Transform(test_set, test_transform)

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    return train_loader, valid_loader, test_loader