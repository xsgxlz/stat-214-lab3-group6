# %%
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import pickle
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel
import matplotlib.pyplot as plt

sys.path.append('code')
sys.path.append("/jet/home/azhang19/stat 214/stat-214-lab3-group6/code")

from BERT.data import TextDataset
from BERT.train_encoder import Args, parse_args, linear_warmup_cosine_decay_multiplicative
from BERT.encoder import ModelArgs, Transformer

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the base path for data access
data_path = '/ocean/projects/mth240012p/shared/data' # Path where data files are stored

# %%
# %% Load preprocessed word sequences (likely includes words and their timings)
with open(f'{data_path}/raw_text.pkl', 'rb') as file:
    wordseqs = pickle.load(file) # wordseqs is expected to be a dictionary: {story_id: WordSequenceObject}

# %% Get list of story identifiers and split into training and testing sets
# Assumes story data for 'subject2' exists and filenames are story IDs + '.npy'
stories = [i[:-4] for i in os.listdir(f'{data_path}/subject2')] # Extract story IDs from filenames
# Split stories into train and test sets with a fixed random state for reproducibility


# First, use 60% for training and 40% for the remaining data.
train_stories, test_stories = train_test_split(stories, train_size=0.75, random_state=214)

# %%
pretrained_bert = BertModel.from_pretrained("bert-base-uncased")
pretrained_word_embeddings = pretrained_bert.embeddings.word_embeddings

# %%
# Define the arguments
args = parse_args()

print(args, end="\n\n")

# %%
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_text = [" ".join(wordseqs[i].data).strip() for i in train_stories]
train_dataset = TextDataset(train_text, tokenizer, max_len=sys.maxsize) # No limitation. The longest sequence is not too long.
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                         num_workers=0, pin_memory=True)
mean_len = (train_dataset.encodings['input_ids'] != 0).sum(dim=1).float().mean().item()
print(f"Mean length across all training sequences: {mean_len:.2f} tokens")

# %%
transformer_args = ModelArgs(
    dim=args.dim,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    hidden_dim=args.hidden_dim,
    vocab_size=pretrained_word_embeddings.num_embeddings,
    norm_eps=1e-5,
    rope_theta=500000,
    max_seq_len=train_dataset.encodings['input_ids'].size(1),
)

model = Transformer(params=transformer_args, pre_train_embeddings=pretrained_word_embeddings).to(device).train()

# %%
# Training configuration
batch_size = args.batch_size

lr = args.standard_lr * batch_size / len(train_stories) # len(train_stories) is the reference batch size
warmup_steps = args.standard_warmup_steps
epochs = args.standard_epoch

print("Derived Parameters:")
print(f"lr: {lr}")
print(f"warmup_steps: {warmup_steps}")
print(f"epochs: {epochs}")
print(f"grad_clip_max_norm: {args.grad_clip_max_norm}", end="\n\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
            lr_lambda=lambda step: linear_warmup_cosine_decay_multiplicative(step, warmup_steps, epochs, args.min_lr))

scaler = torch.amp.GradScaler(device, enabled=args.use_amp)

# %%
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    (Corrected version with explicit device handling)
    Args:
        input_ids (Tensor): [batch_size, seq_len] tensor of token ids
        vocab_size (int): total number of tokens in vocab
        mask_token_id (int): token id used for [MASK]
        pad_token_id (int): token id used for padding
        mlm_prob (float): probability of masking a token
    Returns:
        masked_input_ids: tensor with some tokens replaced for MLM
        masked_indices: boolean tensor indicating which tokens were masked
    """
    device = input_ids.device  # Get device from input tensor

    # Generate mask: which tokens to mask
    probability_matrix = torch.full(input_ids.shape, mlm_prob, device=device)
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    # Ensure boolean tensor for masked_indices is created
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Replace 80% of the masked tokens with [MASK]
    # Ensure boolean tensor for indices_replaced is created
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    # Clone input_ids before modifying to avoid modifying original if it's needed elsewhere
    masked_input_ids = input_ids.clone()
    masked_input_ids[indices_replaced] = mask_token_id

    # Replace 10% of the masked tokens with random token
    # Ensure boolean tensor for indices_random is created
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced 
    random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=torch.long, device=device)
    masked_input_ids[indices_random] = random_tokens[indices_random]

    # 10% remain unchanged (masked_indices & not replaced & not random)

    return masked_input_ids, masked_indices

# %%
def bert_loss_fn(input_ids, logits, loss_mask):
    '''
    Implement BERT loss function
    Args:
        input_ids: Input IDs (batch_size, seq_len) int
        logits: Model logits (batch_size, seq_len, vocab_size) float
        loss_mask: Mask for whether to include the token in the loss (batch_size, seq_len) bool
    Returns:
        loss: Scalar cross-entropy loss float
    '''
    # get dimensions of logits tensor
    batch_size, seq_len, vocab_size = logits.size()

    # input dimension and type validation
    assert input_ids.size() == (batch_size, seq_len), f"input_ids: expected ({batch_size}, {seq_len}), got {tuple(input_ids.size())}"
    assert loss_mask.size() == (batch_size, seq_len), f"loss_mask: expected ({batch_size}, {seq_len}), got {tuple(loss_mask.size())}"
    assert loss_mask.dtype == torch.bool, f"loss_mask must be boolean, got {loss_mask.dtype}"
    
    # flatten input tensors
    logits = logits.view(-1, vocab_size) # to (batch_size * seq_len, vocab_size)
    input_ids = input_ids.view(-1) # to (batch_size * seq_len)
    loss_mask = loss_mask.view(-1) # to (batch_size * seq_len)

    # use mask to filter only unknown tokens
    # where loss_mask (bool): True -> include in loss 
    logits_masked = logits[loss_mask]
    input_ids_masked = input_ids[loss_mask]

    # compute cross-entropy on unnormalized logits and true class indices
    # Use reduction='sum', normalize later
    loss = torch.nn.functional.cross_entropy(logits_masked, input_ids_masked, reduction='sum')    
    return loss


# %%
def backward_pass(model, loss, optimizer, scaler, grad_clip_max_norm):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
    scaler.step(optimizer)
    scaler.update()

# %%
def train_step(model, input_ids, masked_input_ids, loss_mask, atten_masks, mean_len, optimizer, scaler, args):
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.use_amp):
        pred = model(masked_input_ids, attn_mask=atten_masks)
        
        loss = bert_loss_fn(input_ids, pred, loss_mask)

        # Normalize loss. Make the weight of each token is the same and the scale is invariant to the batch size and mlm_prob
        loss_for_backward = loss / (mean_len * batch_size * args.mlm_prob)

    backward_pass(model, loss_for_backward, optimizer, scaler, args.grad_clip_max_norm)

    return loss.item()

# %%
@torch.compile(disable=not args.use_compile)
def train_one_epoch(model, dataloader, mean_len, optimizer, scheduler, scaler, args):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        atten_masks = batch['attention_mask'].to(device)

        masked_input_ids, loss_mask = mask_tokens(input_ids, pretrained_word_embeddings.num_embeddings,
                                              tokenizer.mask_token_id, tokenizer.pad_token_id, args.mlm_prob)

        loss = train_step(model, input_ids, masked_input_ids, loss_mask, atten_masks, mean_len, optimizer, scaler, args)
        total_loss += loss
    
    scheduler.step()
    return total_loss

# %%
def get_name(args, epoch):
    name = f"dim{args.dim}_mlm{args.mlm_prob}_epoch{epoch}"
    return name

def save_model(model, loss_record, name, args):
    save_path = f"{args.save_path}/{name}.pth"
    torch.save((model, loss_record), save_path)

# %%
ckpt_epoch = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

loss_record = np.zeros(epochs)

epoch = 0

while epoch < epochs:

    t0 = time.time()

    if epoch in ckpt_epoch:
        name = get_name(args, epoch)
        save_model(model, loss_record, name, args)
    
    loss_record[epoch] = train_one_epoch(model, dataloader, mean_len, optimizer, scheduler, scaler, args)
    # Normalize loss with mean number of masked tokens
    loss_record[epoch] = loss_record[epoch] / (len(train_stories) * mean_len * args.mlm_prob)

    epoch = epoch + 1

    print(f"Epoch: {epoch}")
    print(f"Loss: {loss_record[epoch-1]:.4f}")
    print(f"Time: {time.time() - t0:.2f} seconds", end="\n\n")

name = get_name(args, epoch)
save_model(model, loss_record, name, args)
