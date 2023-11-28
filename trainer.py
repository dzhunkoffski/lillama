import torch

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from wandb_logger import WandbLogger
import math

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_steps))
        lr_factor *= min(epoch / self.warmup, 1.0)
        return lr_factor

def training_epoch(model, optimzier, criterion, train_loader, device, tqdm_desc):
    train_loss = 0.0
    model.train()
    for indices, lengths in tqdm(train_loader, desc=tqdm_desc):
        indices = indices.to(device)
        # lenghts = lenghts.to(device)

        optimzier.zero_grad()
        logits = model(indices[:, :-1])
        logits = torch.permute(logits, (0, 2, 1))
        loss = criterion(logits, indices[:, 1:])
        loss.backward()
        optimzier.step()

        train_loss += loss.item() * indices.shape[0]
    
    train_loss /= len(train_loader.dataset)
    return train_loss

@torch.no_grad()
def validation_epoch(model, critetion, val_loader, device, tqdm_desc):
    val_loss = 0.0
    model.eval()
    for indices, lengths in tqdm(val_loader):
        indices = indices.to(device)
        # lengths = lengths.to(device)

        logits = model(indices[:, :-1])
        logits = torch.permute(logits, (0, 2, 1))
        loss = critetion(logits, indices[:, 1:])

        val_loss += loss.item() * indices.shape[0]
    val_loss /= len(val_loader.dataset)
    return val_loss

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, device, logger: WandbLogger, scheduler=None, log_output: bool = False):
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        if log_output:
            print(f'=== Epoch {epoch} ===')
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader, device, tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader, device, tqdm_desc=f'Training {epoch}/{num_epochs}'
        )

        results = {}
        if scheduler is not None:
            scheduler.step()
            results["learning rate"] = scheduler.get_last_lr()[0]
        results["train_loss"] = train_loss
        results["val loss"] = val_loss
        train_losses += [train_loss]
        val_losses += [val_loss]

        logger.log_metrics(results)

        if log_output:
            print('Train loss:', train_losses[-1])
            print('Val loss:', val_losses[-1])
