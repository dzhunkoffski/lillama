from src.dataset.dataset import TextDataset
import torch 
from torch import nn

data = TextDataset(
    corpus_path = '/kaggle/input/lillama-corpus/corpus.txt',
    json_path='data/tiny_stories',
    save_tokenizer_to = 'bpe',
    max_len = 256,
    vocab_size = 5000
)
print('created train data with size:', len(data))

generator = torch.Generator().manual_seed(42)
datasets = torch.utils.data.random_split(data, [0.95, 0.05], generator=generator)
train_data = datasets[0]
test_data = datasets[1]
print(len(train_data))
print(len(test_data))

from torch.utils.data import DataLoader
from src.model.language_model import LanguageModel
from src.dataset.dataset import Collator
from src.utils.trainer import CosineAnnealingWithWarmupLR

collate_fn = Collator(pad_value=data.pad_id)
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=4, collate_fn=collate_fn)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LanguageModel(
    embed_dim=64,
    vocab_size=data.vocab_size,
    max_len=data.max_len,
    pad_idx=data.pad_id,
    num_layers=2,
    num_heads=2,
    dropout=0.1,
    feedforward_dim=64
)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
lr_scheduler = CosineAnnealingWithWarmupLR(optimizer=optimizer, warmup_steps=5, max_steps=100)

print(model)

from src.utils.wandb_logger import WandbLogger
from src.utils.trainer import train

wdb = WandbLogger(
    project_name="little-lama",
    config={"loh": "loh"}
)
train(
    model, optimizer, criterion, train_loader, 
    train_loader, 100, DEVICE, wdb, 
    log_output = False, grad_clipping=10, scheduler=lr_scheduler
)

