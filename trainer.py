import torch
from torch import nn
from torch import Tensor

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from wandb_logger import WandbLogger
import math

import wandb

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

        logger.set_step((epoch) * len(train_loader.dataset))
        if scheduler is not None:
            scheduler.step()
            logger.log_state("learning_rate", scheduler.get_last_lr()[0])
        
        logger.log_state("train_loss", train_loss)
        logger.log_state("val_loss", val_loss)
        if isinstance(train_loader.dataset, 'torch.utils.data.Subset'):
            tokenizer = train_loader.dataset.dataset.tokenizer
            bos_id = train_loader.dataset.dataset.bos_id
            vocab_sz = train_loader.dataset.dataset.vocab_size
        else:
            tokenizer = train_loader.dataset.tokenizer
            bos_id = train_loader.dataset.bos_id
            vocab_sz = train_loader.dataset.vocab_size
        text_table = log_predictions(
            model, tokenizer, batch_size=10, device=device, token_len=100, 
            prefix=torch.tensor([bos_id]), vocab_size=vocab_sz
        )
        logger.log_state('text_table', text_table)
        train_losses += [train_loss]
        val_losses += [val_loss]

def log_predictions(model, tokenizer, batch_size: int, device, token_len: int, prefix: Tensor, vocab_size: int):
    generated_ids = generate_nucleus(
        model, tokenizer, batch_size=batch_size, device=device, prefix=prefix, max_len=token_len, nucleus=0.9, vocab_size=vocab_size
    )
    texts = []
    for batch_ix in range(batch_size):
        text_id = generated_ids[batch_ix].cpu().tolist()
        texts.append([tokenizer.decode_ids(text_id)])
    return wandb.Table(data=texts, columns=['Story example'])

@torch.no_grad()
def generate_nucleus(model, tokenizer, batch_size: int, device, prefix: Tensor = None, max_len=100, nucleus=0.9, vocab_size: int = 1000):
    """
    Samples output sequence from probability distribution obtained by model

    :params
        model: predict next token for the whole batch of sequences
        tokenizer: tokenizer for the model and [BOS] token
        batch_size: number of sequence
        prefix: Tensor of tokens with shape: [batch_size, seq_len]
        max_len: max length of predicted sequence
        nucleus: parameter of nucleus sampling

    :return
        the Tensor of tokens of shape: [batch_size, max_len]
    """
    
    prefix = torch.stack(batch_size * [prefix], dim=0)
    prefix = prefix.to(device)

    for i in range(max_len):
        next_token = model.get_next_token(prefix)
        ordered_probs, ordered_indexes = torch.topk(next_token, vocab_size, largest=True, sorted=True)
        probs_cumsum = torch.cumsum(ordered_probs, dim=-1)
        tokens = []
        for batch_ix in range(batch_size):
            ordered_ixs_threshold = torch.argwhere(probs_cumsum[batch_ix] > nucleus).squeeze()[0]
            sampled_ix = torch.multinomial(ordered_probs[batch_ix][:ordered_ixs_threshold+1], 1)
            tokens.append(ordered_indexes[batch_ix][sampled_ix])
        tokens = torch.tensor(tokens).unsqueeze(-1).to(device)
        prefix = torch.cat([prefix, tokens], dim=1)

    return prefix
