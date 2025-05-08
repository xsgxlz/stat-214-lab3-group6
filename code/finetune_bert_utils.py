import os
import numpy as np
import torch
import torch.nn as nn


def get_sliding_window_embeddings(
    base_model: nn.Module,
    token_tensor: torch.Tensor,
    atten_mask: torch.Tensor,
    window_size: int = 512,
    stride: int = 256,
    device: str = None
) -> torch.Tensor:
    """
    Applies a sliding window to process long token sequences with a fixed-window model
    and returns aggregated embeddings for the entire sequence.

    Args:
        base_model: The pre-trained BERT-like model.
        token_tensor: Tensor of token IDs for the full sequence.
                      Shape: (batch_size, sequence_length).
        atten_mask: Attention mask for the full sequence.
                    Shape: (batch_size, sequence_length).
        window_size: The maximum sequence length the base_model can handle.
        stride: The step size for the sliding window. Smaller stride means more overlap.
        device: The device to run the model and tensors on (e.g., 'cuda', 'cpu').
                If None, uses the device of token_tensor.

    Returns:
        torch.Tensor: Aggregated embeddings for the entire sequence.
                      Shape: (batch_size, sequence_length, hidden_size).
    """
    if device is None:
        device = token_tensor.device
    
    batch_size, original_seq_len = token_tensor.shape
    hidden_size = base_model.config.hidden_size

    # Initialize tensors to store aggregated embeddings and counts for averaging
    final_embeddings = torch.zeros(batch_size, original_seq_len, hidden_size, device=device)
    counts = torch.zeros(batch_size, original_seq_len, device=device)

    for i in range(0, original_seq_len, stride):
        start_idx = i
        end_idx = min(i + window_size, original_seq_len)
        
        # Ensure the chunk is not empty
        if start_idx >= end_idx:
            continue

        chunk_tokens = token_tensor[:, start_idx:end_idx].to(device)
        chunk_atten_mask = atten_mask[:, start_idx:end_idx].to(device)

        # Get model outputs for the current chunk
        # The `base_model` (BertModel) typically doesn't need token_type_ids if not fine-tuned for NSP
        outputs = base_model(input_ids=chunk_tokens, attention_mask=chunk_atten_mask)
        chunk_embeddings = outputs.last_hidden_state # (batch_size, chunk_len, hidden_size)

        # Add the chunk embeddings to the final_embeddings tensor
        # and update counts for the processed positions
        final_embeddings[:, start_idx:end_idx, :] += chunk_embeddings
        counts[:, start_idx:end_idx] += 1
            

    # Average the embeddings where positions were covered by multiple windows
    # Avoid division by zero for positions that were not covered (if stride > window_size)
    # Though with proper stride <= window_size, all relevant tokens should be covered at least once.
    counts = torch.max(counts, torch.ones_like(counts)) # Replace 0s with 1s to avoid division by zero
    final_embeddings /= counts.unsqueeze(-1) # Add dimension for broadcasting

    return final_embeddings

def lanczosfun_torch(cutoff: float, t, window: int = 3) -> torch.Tensor:
    """
    Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped PyTorch tensor.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    (Torch version)
    """
    if not isinstance(t, torch.Tensor):
        # If t is a Python scalar, convert to a tensor.
        # Default to float32, assuming cutoff is also float-compatible.
        t_tensor = torch.tensor(t, dtype=torch.float32)
    else:
        t_tensor = t

    # Ensure operations are on the correct device and dtype if t is a tensor
    device = t_tensor.device
    # Use a consistent float dtype for calculations, promoting if t_tensor is lower precision like float16
    # For stability, let's use float32 for internal calculations unless t_tensor is float64
    # Original NumPy likely uses float64 by default for such operations.
    # To be "minimal change", if t_tensor is float, use its dtype. If int, promote to float32.
    if t_tensor.is_floating_point():
        calc_dtype = t_tensor.dtype
    else: # if t_tensor is int or bool
        calc_dtype = torch.float32
    
    t_scaled = t_tensor.to(calc_dtype) * torch.as_tensor(cutoff, dtype=calc_dtype, device=device)
    
    pi_val = torch.pi
    window_float = float(window) # Ensure window is float for division

    # Initialize val with the same shape as t_scaled
    val = torch.zeros_like(t_scaled)

    # Condition for t_scaled != 0
    nonzero_mask = t_scaled != 0
    
    # Calculate for non-zero t_scaled
    if torch.any(nonzero_mask):
        t_scaled_nz = t_scaled[nonzero_mask]
        numerator_nz = window_float * torch.sin(pi_val * t_scaled_nz) * \
                       torch.sin(pi_val * t_scaled_nz / window_float)
        denominator_nz = (pi_val**2 * t_scaled_nz**2)
        val[nonzero_mask] = numerator_nz / denominator_nz
    
    # Set val for t_scaled == 0
    # This will correctly handle broadcasting if t_scaled was multi-dimensional
    val[t_scaled == 0] = 1.0 
    
    # Set val for abs(t_scaled) > window
    val[torch.abs(t_scaled) > window_float] = 0.0
    
    # Original code had a commented-out normalization:
    # return val # / (torch.sum(val) + 1e-10) 
    return val


def lanczosinterp2D_torch(
    data: torch.Tensor, 
    oldtime: torch.Tensor, 
    newtime: torch.Tensor, 
    window: int = 3, 
    cutoff_mult: float = 1.0, 
    rectify: bool = False
) -> torch.Tensor:
    """
    Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.
    (Torch version)
    
    Args:
        data (torch.Tensor): Input data tensor of shape (num_old_samples, num_features).
        oldtime (torch.Tensor): 1D tensor of time points for data samples.
        newtime (torch.Tensor): 1D tensor of new time points for interpolation.
    """
    device = data.device
    # Use float32 for time calculations and sincmat for consistency, unless data is float64
    # The original NumPy would default to float64 for np.zeros and calculations.
    # To be "minimal", if data is float64, try to preserve it. Otherwise, float32 is common.
    if data.dtype == torch.float64:
        time_dtype = torch.float64
        sincmat_dtype = torch.float64
    else: # Includes float32, float16, bfloat16, and also non-float types if not careful
        time_dtype = torch.float32
        sincmat_dtype = torch.float32 # sincmat should be float

    oldtime_t = oldtime.to(device=device, dtype=time_dtype)
    newtime_t = newtime.to(device=device, dtype=time_dtype)
    
    ## Find the cutoff frequency ##
    if len(newtime_t) <= 1:
        # Handle edge cases where diff or mean might not be well-defined or lead to div by zero
        if len(newtime_t) == 1 and len(oldtime_t) > 0:
            # If only one new time point, behavior is ill-defined for frequency-based cutoff.
            # Original NumPy would yield nan/warning. Using a placeholder.
            cutoff = torch.tensor(1.0, dtype=time_dtype, device=device) 
        else: # No new time points or insufficient old time points
            num_out_features = data.shape[1] * (2 if rectify else 1)
            return torch.empty((len(newtime_t), num_out_features), device=device, dtype=data.dtype)
    else:
        mean_diff_newtime = torch.mean(torch.diff(newtime_t))
        if mean_diff_newtime == 0: # Avoid division by zero if all newtime points are the same
            cutoff = torch.tensor(1.0, dtype=time_dtype, device=device) # Placeholder
        else:
            cutoff = (1.0 / mean_diff_newtime) * cutoff_mult
    
    ## Build up sinc matrix ##
    # sincmat dtype should be float for matrix multiplication with data (which also should be float)
    sincmat = torch.zeros((len(newtime_t), len(oldtime_t)), dtype=sincmat_dtype, device=device)
    
    for ndi in range(len(newtime_t)):
        time_diffs = newtime_t[ndi] - oldtime_t # Operates on time_dtype tensors
        # lanczosfun_torch will use dtype of time_diffs or promote to float32 if time_diffs is int
        sincmat[ndi, :] = lanczosfun_torch(cutoff.item(), time_diffs, window).to(sincmat_dtype)
    
    # Ensure data is float for matmul with sincmat
    data_float = data.to(sincmat_dtype) if not data.is_floating_point() or data.dtype != sincmat_dtype else data

    if rectify:
        # torch.clip expects min/max to be scalar or tensor. Python float 0.0 is fine.
        data_neg = torch.clip(data_float, max=0.0) 
        data_pos = torch.clip(data_float, min=0.0)
        
        interp_neg = torch.matmul(sincmat, data_neg)
        interp_pos = torch.matmul(sincmat, data_pos)
        newdata = torch.hstack([interp_neg, interp_pos])
    else:
        newdata = torch.matmul(sincmat, data_float)

    return newdata

def downsample_word_vectors_torch(
    stories: list, 
    word_vectors: dict, 
    wordseqs: dict,
    device_str: str = None # e.g., "cuda", "cpu"
) -> dict:
    """
    Get Lanczos downsampled word_vectors for specified stories using PyTorch.

    Args:
        stories: List of stories to obtain vectors for.
        word_vectors: Dictionary of {story: np.ndarray or torch.Tensor [num_story_words, vector_size]}
        wordseqs: Dictionary of {story: object with .data_times and .tr_times (as np.ndarray or torch.Tensor)}.
        device_str: Device string to perform computations on. If None, uses CUDA if available, else CPU.

    Returns:
        Dictionary of {story: downsampled vectors as torch.Tensor}
    """
    if device_str is None:
        computed_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        computed_device = torch.device(device_str)
    
    downsampled_semanticseqs_torch = dict()
    for story in stories:
        # Convert word_vectors[story] to torch.Tensor if it's NumPy
        wv_data = word_vectors[story]
        if isinstance(wv_data, np.ndarray):
            # Original type hint suggests float32
            wv_tensor = torch.from_numpy(wv_data).float().to(computed_device)
        elif isinstance(wv_data, torch.Tensor):
            wv_tensor = wv_data.to(computed_device)
        else:
            raise TypeError(f"word_vectors for story '{story}' must be np.ndarray or torch.Tensor")

        # Convert .data_times and .tr_times to torch.Tensor if they are NumPy
        dt_data = wordseqs[story].data_times
        tt_data = wordseqs[story].tr_times

        if isinstance(dt_data, np.ndarray):
            data_times_tensor = torch.from_numpy(dt_data).float().to(computed_device)
        elif isinstance(dt_data, torch.Tensor):
            data_times_tensor = dt_data.to(computed_device)
        else:
            raise TypeError(f"wordseqs.data_times for story '{story}' must be np.ndarray or torch.Tensor")

        if isinstance(tt_data, np.ndarray):
            tr_times_tensor = torch.from_numpy(tt_data).float().to(computed_device)
        elif isinstance(tt_data, torch.Tensor):
            tr_times_tensor = tt_data.to(computed_device)
        else:
            raise TypeError(f"wordseqs.tr_times for story '{story}' must be np.ndarray or torch.Tensor")
            
        downsampled_semanticseqs_torch[story] = lanczosinterp2D_torch(
            wv_tensor, 
            data_times_tensor, 
            tr_times_tensor, 
            window=3
        )
    return downsampled_semanticseqs_torch

def aggregate_embeddings(story_embeddings, stories):
    """
    Aggregate the embeddings of the stories into a single tensor.
    """
    all_embeddings = []
    for story in stories:
        all_embeddings.append(story_embeddings[story])
    return torch.cat(all_embeddings, dim=0)

def load_fmri_data(stories, data_path):
    """
    Loads fMRI data (.npy files) for given stories and subjects.

    Args:
        stories (list): List of story identifiers to load fMRI data for.
        data_path (str): Base path where subject fMRI data is stored (e.g., data_path/subject_id/story_id.npy).

    Returns:
        dict: A nested dictionary `{subject_id: {story_id: fmri_data_array}}`.
    """
    subjects = ['subject2', 'subject3'] # List of subjects to load data for
    fmri_data = {} # Outer dictionary {subject: {story: data}}
    for subject in subjects:
        subject_dict = {} # Inner dictionary {story: data} for the current subject
        for story in stories:
            # Construct the full path to the fMRI data file
            file_path = os.path.join(data_path, subject, f'{story}.npy')
            # Load the NumPy array from the file
            data = np.load(file_path)
            subject_dict[story] = data # Store data in the inner dictionary
        fmri_data[subject] = subject_dict # Store the subject's data dictionary
    return fmri_data

def get_fmri_data(stories, fmri_data):
    """
    Concatenates fMRI data across specified stories for each subject.
    Returns a dictionary: {subject: concatenated_fmri_array}.
    """
    out_dict = {}
    for subj in fmri_data.keys():
        concatenated = np.concatenate(
            [fmri_data[subj][st] for st in stories], axis=0
        )
        out_dict[subj] = concatenated
    return out_dict