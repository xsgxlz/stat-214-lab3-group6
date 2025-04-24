import math
from dataclasses import dataclass

import torch
# --- MLM MASKING ---
import torch

def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    Args:
        input_ids (Tensor): [batch_size, seq_len] tensor of token ids
        vocab_size (int): total number of tokens in vocab
        mask_token_id (int): token id used for [MASK]
        pad_token_id (int): token id used for padding
        mlm_prob (float): probability of masking a token
    Returns:
        masked_input_ids: tensor with some tokens replaced for MLM
        labels: tensor with -100 (ignore index) except masked tokens with original id
    """
    labels = input_ids.clone()
    # Generate mask: which tokens to mask
    probability_matrix = torch.full(input_ids.shape, mlm_prob)
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Set labels to -100 for non-masked positions so loss is ignored
    labels[~masked_indices] = -100

    # Replace 80% of the masked tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # Replace 10% of the masked tokens with random token
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% remain unchanged (masked_indices & not replaced & not random)
    return input_ids, labels


def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    '''
    TODO: Implement training loop for BERT
    Args:
        model: BERT model
        dataloader: Data loader
        tokenizer: Tokenizer
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run the model on
    '''
    pass

@dataclass(frozen=True)
class Args:
    # Training parameters
    standard_lr: float = 2.5e-4
    standard_epoch: int = 20000
    standard_warmup_steps: int = 4000
    batch_size: int = 256
    min_lr: float = 1e-4
    grad_clip_max_norm: float = 5.0
    use_amp: bool = False
    use_compile: bool = False

    # Model architecture parameters
    dim: int = 24
    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 84

    # Save path
    save_path: str = ""
    final_save_path: str = ""

    def __str__(self):
        """Return a formatted string representation of the simplified Args object."""
        # Define groups of parameters based on remaining attributes
        training_params = [
            f"standard_lr:        {self.standard_lr:.1e}",
            f"standard_epoch:     {self.standard_epoch}",
            f"standard_warmup_steps: {self.standard_warmup_steps}",
            f"batch_size:         {self.batch_size}",
            f"min_lr:             {self.min_lr:.1e}",
            f"grad_clip_max_norm: {self.grad_clip_max_norm:.1f}",
            f"use_amp:            {self.use_amp}",
            f"use_compile:        {self.use_compile}",
        ]

        model_params = [
            f"dim:               {self.dim}",
            f"n_layers:          {self.n_layers}",
            f"n_heads:           {self.n_heads}",
            f"hidden_dim:        {self.hidden_dim}",
        ]

        save_params = [
            f"save_path:         {self.save_path}",
            f"final_save_path:   {self.final_save_path}",
        ]

        # Combine sections with headers
        sections = [
            ("Training Parameters", training_params),
            ("Model Architecture Parameters", model_params),
            ("Save Path Parameters", save_params),
        ]

        # Build the formatted string
        output = "Args Configuration:\n"
        for header, params in sections:
            # No conditional logic needed as only relevant sections remain
            output += f"\n{header}:\n"
            output += "\n".join([f"  {param}" for param in params])
            output += "\n"

        return output.rstrip()  # Remove trailing newline
    
def linear_warmup_cosine_decay(current_step: int, warmup_steps: int, total_steps: int, min_lr: float):
    if current_step < warmup_steps:
        # Linear warm-up
        return float(current_step + 1) / float(warmup_steps + 1)
    # Cosine decay
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) + min_lr

def linear_warmup_cosine_decay_multiplicative(current_step: int, warmup_steps: int, total_steps: int, min_lr: float):
    if current_step != 1:
        pre = linear_warmup_cosine_decay(current_step - 1, warmup_steps, total_steps, min_lr)
        now = linear_warmup_cosine_decay(current_step, warmup_steps, total_steps, min_lr)
        return now / pre
    else:
        return linear_warmup_cosine_decay(1, warmup_steps, total_steps, min_lr)
