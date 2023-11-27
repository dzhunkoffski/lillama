import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from src.utils.wandb_logger import WandbLogger
from src.model.samplers import generate_nucleus
import math

import wandb

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()

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

def training_epoch(model, optimzier, criterion, train_loader, device, epoch, logger, grad_clipping):
    train_loss = 0.0
    model.train()
    for i, (indices, lengths) in tqdm(list(enumerate(train_loader))):
        logger.set_step(step=(epoch-1) * len(train_loader.dataset) + i)
        indices = indices.to(device)
        # lenghts = lenghts.to(device)

        optimzier.zero_grad()
        logits = model(indices[:, :-1])
        logits = torch.permute(logits, (0, 2, 1))
        loss = criterion(logits, indices[:, 1:])
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clipping)
        optimzier.step()

        logger.log_state('train_loss', loss.item())
        logger.log_state('grad_norm', get_grad_norm(model))
        train_loss += loss.item() * indices.shape[0]
    
    train_loss /= len(train_loader.dataset)
    return train_loss

@torch.no_grad()
def validation_epoch(model, critetion, val_loader, device, epoch):
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

def train(
        model, optimizer, criterion, train_loader, val_loader, 
        num_epochs, device, logger: WandbLogger, grad_clipping: float = 1000.0, 
        scheduler=None, log_output: bool = False):
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        print(f'=== Epoch {epoch} ===')
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader, device, epoch=epoch, logger=logger, grad_clipping=grad_clipping
        )
        val_loss = validation_epoch(
            model, criterion, val_loader, device, epoch=epoch
        )
        if scheduler is not None:
            scheduler.step()
            logger.log_state("lr", scheduler.get_last_lr()[0])

        logger.log_state('val_loss', val_loss)
        text_table = log_predictions(
            model, train_loader.dataset.tokenizer, batch_size=10, device=device, 
            token_len=100, prefix=torch.tensor([train_loader.dataset.bos_id]),
            vocab_size=train_loader.dataset.vocab_size
        )
        logger.log_state('text_table', text_table)
    
def log_predictions(model, tokenizer, batch_size: int, device, token_len: int, prefix: Tensor, vocab_size: int):
    generated_ids = generate_nucleus(
        model, tokenizer, batch_size=batch_size, 
        device=device, prefix=prefix, max_len=token_len, nucleus=0.9, vocab_size=vocab_size
    )
    texts = []
    for batch_ix in range(batch_size):
        text_id = generated_ids[batch_ix].cpu().tolist()
        texts.append([tokenizer.decode_ids(text_id)])
    return wandb.Table(data=texts, columns=['Story example'])

    