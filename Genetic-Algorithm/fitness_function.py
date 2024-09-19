import torch
from torch import nn

from chromosome import Chromosome
from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from model import MLP

# intervals for the 3 parameters that we want to optimize
n_interval = [1, 120]
lr_interval = [1, 1e-5]
wd_interval = [1, 1e-5]

num_epochs = 100
batch_size = 8

train_dataset = CustomDataset('data/wine+quality/train-red.csv')
val_dataset = CustomDataset('data/wine+quality/val-red.csv')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


def evaluate_fitness(chromosome: Chromosome):
    third_length = Chromosome.length // 3

    chromosome.n = int(binary_to_decimal_in_range(chromosome.genes[:third_length],
                                                  n_interval[0], n_interval[1]))

    chromosome.lr = binary_to_decimal_in_range(chromosome.genes[third_length:2 * third_length],
                                               lr_interval[0], lr_interval[1])

    chromosome.wd = binary_to_decimal_in_range(chromosome.genes[2 * third_length:],
                                               wd_interval[0], wd_interval[1])

    mlp = MLP([11, chromosome.n, 1])

    lr = chromosome.lr
    weight_decay = chromosome.wd

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    mlp.train()
    for epoch in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_targets = mlp(features)

            loss = criterion(predicted_targets, targets)
            loss.backward()
            optimizer.step()

    val_loss = 0

    mlp.eval()
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(val_loader):
            predicted_targets = mlp(features)
            loss = criterion(predicted_targets, targets)
            val_loss += loss.item() * targets.size(0)

    val_loss /= len(val_dataset)

    chromosome.fitness_value = 1 / (val_loss + 1e-10)


def binary_to_decimal_in_range(value, lower, upper):
    decimal_value = sum([gene * (2**power) for gene, power in zip(value, range(len(value) - 1, -1, -1))])
    decimal_value = lower + decimal_value * ((upper - lower) / (2**len(value) - 1))
    return decimal_value
