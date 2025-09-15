import torch, os, random
import numpy as np
from torch import optim, nn
from Basic_Resnet18 import resnet18_cifar as res_net18
from torch.utils.tensorboard import SummaryWriter

import data_pre

# set parameters
config = {
    'seed': 42,
    'batch_size': 128,
    'learning_rate': 0.1,
    'n_epochs': 200,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_classes': 10,
    'num_workers': 2,
    'early_stop': 20,
    'valid_ratio': 0.2,
}

# set random seed
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def resnet18_cifar_model(num_classes=10):
    return res_net18(num_classes=num_classes)

def train(train_loader, valid_loader, model, config, device, logdir='runs/resnet18_cifar10'):
    writer = SummaryWriter(log_dir=logdir)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # decay lr
    num_epochs = config['n_epochs']
    best_acc = 0.0
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (outputs.argmax(1) == labels).sum().item()
            epoch_samples += labels.size(0)

            # if (i + 1) % 100 == 0:  # print every 100 mini_batches
            #     avg_loss = running_loss / 100
            #     avg_acc = running_correct / running_samples * 100
            #     print(f'[Epoch {epoch + 1}, Iter {i + 1}] loss: {avg_loss:.4f}, Accuracy:{avg_acc:.2f}%')
            #     running_loss = 0.0
            #     running_samples = 0
            #     running_correct = 0

        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples * 100.0
        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch+1)

        # validation
        val_loss, val_acc = evaluate_loader(valid_loader, model, device, criterion)
        print(f'Epoch {epoch+1}: val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.2f}%')
        writer.add_scalar('Valid/Loss', val_loss, epoch+1)
        writer.add_scalar('Valid/Accuracy', val_acc, epoch+1)

        # early stop
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            # save the best model
            # save_model(model, 'best_model.pth')
        else:
            patience += 1
            if patience >= config['early_stop']:
                print(f'Early stopping at epoch {epoch}')
                break

        scheduler.step()
    writer.close()

# evaluate valid_set when training
def evaluate_loader(loader, model, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            samples += labels.size(0)

    avg_loss = total_loss / samples
    avg_acc = correct / samples * 100
    return avg_loss, avg_acc

# evaluate test_set after training
def evaluate(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nTest Accuracy: {100 * correct / total:.2f}%')

def save_model(model, file_name='resnet18_cifar10.pth'):
    base_dir = os.path.dirname(__file__) # get current file directory
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, file_name)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == '__main__':
    set_seed(config['seed'])
    train_loader, valid_loader, test_loader = data_pre.get_dataloaders(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet18_cifar_model(num_classes=config['num_classes']).to(device)
    train(train_loader, valid_loader, model, config, device)

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth'), map_location=device))
    model_best = model.to(device)
    evaluate(test_loader, model_best, device)

    # save the model
    # save_model(model, file_name='resnet18_cifar10_valid.pth')
    # evaluate(test_loader, model, device)
