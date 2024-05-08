import torch
import torchvision

from model import Model
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', '-m')
    parser.add_argument('--dataset', '-d', choices=['mnist', 'fashion-mnist'])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--criterion', default='cross_entropy')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()

    # Device
    use_gpu = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(torch.flatten)
    ])
    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)
        input_size = 28 ** 2
        output_size = 10

    if use_gpu:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    else:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    hidden_sizes = [int(layer_size) for layer_size in args.model.split('_')]
    model = Model(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size).to(device)

    # Criterion
    if args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Log
    writer = SummaryWriter()

    # Train
    for epoch in tqdm(range(args.num_epochs), desc='Epoch'):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_data_loader:
            optimizer.zero_grad()
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_data_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_data_loader:
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_data_loader.dataset)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)


if __name__ == '__main__':
    main()