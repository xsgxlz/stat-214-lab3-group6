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
    loss = torch.nn.functional.cross_entropy(logits_masked, input_ids_masked)
    
    return loss
