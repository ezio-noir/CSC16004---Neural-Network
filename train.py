import os

import torch
import torchvision

from model import get_model_by_id, init_model_with_weight
from features import extract_hog_128
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=int)
    parser.add_argument('--load-weight')
    parser.add_argument('--dataset', '-d', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--use-feature', choices=['hog'])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--criterion', default='cross_entropy')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    args = parser.parse_args()

    # Device
    use_gpu = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Dataset and dataloader
    if args.use_feature:
        if args.use_feature == 'hog':
            transform = transforms.Compose([
                transforms.Lambda(extract_hog_128),
                transforms.Lambda(torch.from_numpy)
            ])
            input_size = 34020
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(torch.flatten)
        ])
        input_size = 28 ** 2
    output_size = 10

    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    elif args.dataset == 'fashion_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True, transform=transform)

    if use_gpu:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    else:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    if args.model:
        model = get_model_by_id(args.model, input_size, output_size).to(device)
        start_epoch = 1
    elif args.load_weight:
        model, start_epoch = init_model_with_weight(os.load_weight)

    # Criterion
    if args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Log
    if not os.path.isdir('./runs'):
        os.mkdir('./runs')
    writer = SummaryWriter()

    # Save model weights
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    min_test_loss = 1e9
    max_test_acc = 0.0

    # Train
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_data_loader, desc=f'Epoch {epoch}'):
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
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_data_loader:
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        test_loss /= len(test_data_loader.dataset)
        accuracy = correct / total

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

        if (epoch) % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), f'./checkpoints/{args.model}_{epoch}.pt')
        if min_test_loss >= test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), f'./checkpoints/{args.model}_{epoch}_loss.pt')
        if max_test_acc <= accuracy:
            max_test_acc = accuracy
            torch.save(model.state_dict(), f'./checkpoints/{args.model}_{epoch}_acc.pt')


if __name__ == '__main__':
    main()