import argparse
import math
from dataclasses import dataclass

import torch
# --- MLM MASKING ---
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
        labels: tensor with -100 (ignore index) except masked tokens with original id
    """
    device = input_ids.device  # Get device from input tensor
    labels = input_ids.clone()

    # Generate mask: which tokens to mask
    probability_matrix = torch.full(input_ids.shape, mlm_prob, device=device)
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    # Ensure boolean tensor for masked_indices is created
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels to -100 for non-masked positions so loss is ignored
    labels[~masked_indices] = -100 # -100 is the ignore index for CrossEntropyLoss

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

    return masked_input_ids, labels

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

    # BERT parameters
    mlm_prob: float = 0.15

    # Save path
    save_path: str = ""

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
        ]

        bert_params = [
            f"mlm_prob:          {self.mlm_prob:.2f}",
        ]

        # Combine sections with headers
        sections = [
            ("Training Parameters", training_params),
            ("Model Architecture Parameters", model_params),
            ("BERT Parameters", bert_params),
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
    
def parse_args() -> Args:
    """Parses command-line arguments and returns an Args object."""

    # Define the default arguments using the provided structure
    default_args = Args(
        # Training
        standard_lr=1e-3,
        standard_epoch=1000,
        standard_warmup_steps=50,
        batch_size=25,
        min_lr=1e-4,
        grad_clip_max_norm=1.0,
        use_amp=True,
        use_compile=True,
        # Model
        dim=32,
        n_layers=2,
        n_heads=4,
        hidden_dim=112,
        # BERT parameters
        mlm_prob = 0.15,
        # Save
        save_path="/jet/home/azhang19/stat 214/stat-214-lab3-group6/code/ckpts",
    )

    parser = argparse.ArgumentParser(description="Train a Transformer Model with MLM.")

    # --- Training Parameters ---
    parser.add_argument('--standard-lr', type=float, default=default_args.standard_lr,
                        help=f'Standard learning rate (default: {default_args.standard_lr})')
    parser.add_argument('--standard-epoch', type=int, default=default_args.standard_epoch,
                        help=f'Number of training epochs (default: {default_args.standard_epoch})')
    parser.add_argument('--standard-warmup-steps', type=int, default=default_args.standard_warmup_steps,
                        help=f'Number of warmup steps for lr scheduler (default: {default_args.standard_warmup_steps})')
    parser.add_argument('--batch-size', type=int, default=default_args.batch_size,
                        help=f'Batch size for training (default: {default_args.batch_size})')
    parser.add_argument('--min-lr', type=float, default=default_args.min_lr,
                        help=f'Minimum learning rate for cosine decay (default: {default_args.min_lr})')
    parser.add_argument('--grad-clip-max-norm', type=float, default=default_args.grad_clip_max_norm,
                        help=f'Maximum norm for gradient clipping (default: {default_args.grad_clip_max_norm})')
    # For booleans where the default is True, use BooleanOptionalAction (requires Python 3.9+)
    # This creates both --use-amp and --no-use-amp
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=default_args.use_amp,
                        help=f'Enable Automatic Mixed Precision (default: {default_args.use_amp})')
    parser.add_argument('--use-compile', action=argparse.BooleanOptionalAction, default=default_args.use_compile,
                        help=f'Enable torch.compile (default: {default_args.use_compile})')

    # --- Model Architecture Parameters ---
    parser.add_argument('--dim', type=int, default=default_args.dim,
                        help=f'Embedding dimension (default: {default_args.dim})')
    parser.add_argument('--n-layers', type=int, default=default_args.n_layers,
                        help=f'Number of transformer layers (default: {default_args.n_layers})')
    parser.add_argument('--n-heads', type=int, default=default_args.n_heads,
                        help=f'Number of attention heads (default: {default_args.n_heads})')
    parser.add_argument('--hidden-dim', type=int, default=default_args.hidden_dim,
                        help=f'Hidden dimension in FFN (default: {default_args.hidden_dim})')

    # --- BERT Parameters ---
    parser.add_argument('--mlm-prob', type=float, default=default_args.mlm_prob,
                        help=f'Probability of masking a token for MLM (default: {default_args.mlm_prob})')

    # --- Save Path ---
    parser.add_argument('--save-path', type=str, default=default_args.save_path,
                        help=f'Directory to save checkpoints (default: "{default_args.save_path}")')


    # Parse arguments from the command line
    parsed_args = parser.parse_args()

    # Create and return an Args object from the parsed arguments
    # Use vars() to convert the argparse namespace to a dictionary
    final_args = Args(**vars(parsed_args))

    return final_args
    
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
