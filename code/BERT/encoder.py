from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    """Configuration class for Transformer model hyperparameters."""
    dim: int = 4096           # Dimensionality of the model (embedding size)
    n_layers: int = 32        # Number of transformer layers
    n_heads: int = 32         # Number of attention heads in each layer
    n_kv_heads: Optional[int] = None  # Number of key/value heads (defaults to n_heads if None)
    hidden_dim: int = 14436   # Size of the hidden layer in the feed-forward network
    vocab_size: int = -1      # Vocabulary size (must be set before use)
    norm_eps: float = 1e-5    # Epsilon value for numerical stability in RMS normalization
    rope_theta: float = 500000  # Base frequency for RoPE (Rotary Position Embedding)
    max_seq_len: int = 2048   # Maximum sequence length for precomputing RoPE frequencies

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

class SelfMultiHeadAttention(nn.Module):
    """
    Computes multi-head attention with optional Grouped Query Attention (GQA)
    and Rotary Positional Embeddings. Uses a single packed projection layer.
    Supports nested or padded tensors.

    Args:
        dim (int): Total embedding dimension.
        n_heads (int): Number of query heads.
        n_kv_heads (Optional[int]): Number of key/value heads for GQA.
            If None, defaults to n_heads (standard MHA). Default: None.
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input/output projections. Default: False
        rope (RotaryPositionalEmbeddings, optional): Initialized RoPE module. Default: None
        device (torch.device, optional): Device for tensors. Default: None
        dtype (torch.dtype, optional): Data type for tensors. Default: None
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias=False,
        rope: Optional[RotaryPositionalEmbeddings] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        # GQA setup
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert dim % n_heads == 0, "Query embedding dim must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = dim // n_heads
        self.bias = bias
        self.is_gqa = self.n_heads != self.n_kv_heads # Flag to enable GQA in SDPA

        # Calculate the total dimension needed for the packed projection
        # Q uses dim, K and V use n_kv_heads * head_dim each
        self.q_size = dim
        self.kv_size = self.n_kv_heads * self.head_dim
        self.packed_proj_size = self.q_size + 2 * self.kv_size

        # Define the single packed projection layer
        self.packed_proj = nn.Linear(dim, self.packed_proj_size, bias=bias, **factory_kwargs)

        # Output projection remains the same
        self.out_proj = nn.Linear(dim, dim, bias=bias, **factory_kwargs)

        # Store the RoPE module instance
        self.rope = rope
        if self.rope is not None:
            # Ensure RoPE dimension matches the head dimension
            assert self.rope.dim == self.head_dim, \
                f"RoPE dimension ({self.rope.dim}) does not match head dimension ({self.head_dim})"


    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        is_causal=True,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply packed input projection
            2. Split projected tensor into Q, K, V based on GQA sizes
            3. Reshape Q, K, V to split heads
            4. Apply RoPE (if available) to Q and K
            5. Prepare for SDPA (transpose)
            6. Run SDPA (enabling GQA if applicable)
            7. Combine heads and apply output projection

        Args:
            x (torch.Tensor): input tensor of shape (N, L, E)
            input_pos (Optional[torch.Tensor]): Optional tensor containing position ids
                for RoPE. Shape (N, L) or (1, L) for training/full sequence,
                or (N, 1) or (1, 1) for inference. Default: None
            is_causal (bool, optional): Whether to apply causal mask in SDPA. Default: True

        Returns:
            attn_output (torch.Tensor): output of shape (N, L, E)
        """
        # x shape: (N, L, dim)

        # Step 1. Apply packed input projection
        # result shape: (N, L, packed_proj_size)
        result = self.packed_proj(x)

        # Step 2. Split projected tensor into Q, K, V
        qkv_size = [self.q_size, self.kv_size, self.kv_size]
        query, key, value = torch.split(result, qkv_size, dim=-1)
        # Shapes: query (N, L, dim), key (N, L, kv_size), value (N, L, kv_size)

        # Step 3. Reshape Q, K, V to split heads
        # Query shape: (N, L, n_heads, head_dim)
        query = query.unflatten(-1, [self.n_heads, self.head_dim])
        # Key/Value shape: (N, L, n_kv_heads, head_dim)
        key = key.unflatten(-1, [self.n_kv_heads, self.head_dim])
        value = value.unflatten(-1, [self.n_kv_heads, self.head_dim])

        # Step 4: Apply RoPE *before* transposing for SDPA, if available
        # RoPE module expects input shape (b, s, n_h, h_d)
        if self.rope is not None:
            query = self.rope(query, input_pos=input_pos)
            key = self.rope(key, input_pos=input_pos)
            # Value is not rotated

        # Step 5. Prepare for SDPA (transpose)
        # Query shape: (N, n_heads, L, head_dim)
        query = query.transpose(1, 2)
        # Key/Value shape: (N, n_kv_heads, L, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Step 6. Run SDPA
        # Input shapes: query (N, n_heads, L, E), key (N, n_kv_heads, S, E), value (N, n_kv_heads, S, Ev)
        # Output shape: (N, n_heads, L, Ev) where Ev is head_dim here
        attn_output = F.scaled_dot_product_attention(
            query,                      # (N, n_heads, L, head_dim)
            key,                        # (N, n_kv_heads, L, head_dim)
            value,                      # (N, n_kv_heads, L, head_dim)
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0, # Apply dropout only during training
            is_causal=is_causal,
            enable_gqa=self.is_gqa       # Enable GQA based on head counts
        )

        # Step 7. Combine heads and apply output projection
        # (N, n_heads, L, head_dim) -> (N, L, n_heads, head_dim) -> (N, L, dim)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        # (N, L, dim) -> (N, L, dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class FeedForward(nn.Module):
    """Feed-forward network with gated activation (e.g., SwiGLU variant)."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        """
        Initialize FeedForward module.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            hidden_dim (int): Hidden layer dimension.
            bias (bool): Whether to include bias in linear layers. Defaults to False.
        """
        super().__init__()
        self.w13 = nn.Linear(in_dim, hidden_dim * 2, bias) # Packed projection
        self.w2 = nn.Linear(hidden_dim, out_dim, bias)  # Output projection

    def forward(self, x):
        """Apply gated feed-forward computation: w2(silu(w1(x)) * w3(x))."""
        x1, x3 = self.w13(x).chunk(2, dim=-1)  # Split into two parts
        return self.w2(F.silu(x1) * x3)  # Apply gated activation and output projection

class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward layers."""
    def __init__(self, args: ModelArgs, rope: Optional[RotaryPositionalEmbeddings] = None):
        """
        Initialize TransformerBlock.

        Args:
            args (ModelArgs): Model configuration parameters.
            rope (Optional[RotaryPositionalEmbeddings]): RoPE module instance shared across layers.
        """
        super().__init__()
        # Initialize attention layer, passing the RoPE module instance
        self.attention = SelfMultiHeadAttention(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            bias=False,  # Typically False in LLaMA variants
            dropout=0.0, # Dropout usually applied elsewhere or not at all in base blocks
            rope=rope,   # Pass RoPE module here
        )
        # Initialize feed-forward layer
        self.feed_forward = FeedForward(
            in_dim=args.dim,
            out_dim=args.dim,
            hidden_dim=args.hidden_dim,
            bias=False, # Typically False in LLaMA variants
        )
        # Initialize RMSNorm layers
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        # Note: input_pos for RoPE is handled implicitly or needs to be passed
        # if SelfMultiHeadAttention requires it dynamically per block.
        # Currently, SelfMultiHeadAttention uses input_pos=None by default.
    ):
        """
        Forward pass for the Transformer block. Applies self-attention and feed-forward network
        with pre-normalization and residual connections. RoPE and causal masking are handled
        within the SelfMultiHeadAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seqlen, dim).

        Returns:
            torch.Tensor: Output tensor of shape (bsz, seqlen, dim).
        """
        # Apply attention with pre-normalization and residual connection
        # Causal masking and RoPE are handled inside self.attention
        h = x + self.attention(self.attention_norm(x), is_causal=True) # input_pos=None implicit

        # Apply feed-forward with pre-normalization and residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    """Full Transformer model with multiple layers."""
    def __init__(self, params: ModelArgs):
        """
        Initialize the Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Token embedding layer
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim) # padding_idx often set based on tokenizer

        # Initialize RoPE module once, to be shared by all layers
        self.rope = RotaryPositionalEmbeddings(
            dim=params.dim // params.n_heads,  # RoPE operates on head dimension
            max_seq_len=params.max_seq_len,
            base=params.rope_theta,
        )

        # Stack of transformer blocks, each receiving the shared RoPE module
        self.layers = nn.ModuleList([TransformerBlock(params, self.rope) for _ in range(params.n_layers)])

        # Final normalization layer
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)

        # Output projection to vocabulary logits
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def update_max_seq_len(self, max_seq_len: int):
        """Updates the RoPE cache for a potentially new max sequence length."""
        self.rope.build_rope_cache(max_seq_len)
        self.params.max_seq_len = max_seq_len # Update stored max_seq_len
        return self

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for the Transformer model.

        Processes input token sequence through embeddings, transformer layers,
        final normalization, and output projection. RoPE and causal masking
        are handled within the attention mechanism of each TransformerBlock.

        Args:
            tokens (torch.Tensor): Input token indices of shape (bsz, seqlen).

        Returns:
            torch.Tensor: Logits of shape (bsz, seqlen, vocab_size).
        """
        _bsz, seqlen = tokens.shape
        # Ensure RoPE cache covers current sequence length during training/eval
        # (Can be handled by update_max_seq_len or ensuring initial max_seq_len is sufficient)
        if seqlen > self.params.max_seq_len:
             self.update_max_seq_len(seqlen) # Dynamically extend if needed

        h = self.tok_embeddings(tokens)  # Convert tokens to embeddings (bsz, seqlen, dim)

        # Pass through transformer layers sequentially
        # RoPE (using default sequential positions) and causal masking are handled within each block's attention
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)  # Apply final normalization (bsz, seqlen, dim)
        output = self.output(h)  # Project to vocabulary logits (bsz, seqlen, vocab_size)

        # Return logits, usually float32 for stability with loss functions
        return output