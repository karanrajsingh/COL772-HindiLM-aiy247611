# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed

# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

import torch
import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import math

device = None
tokenizer = None

def evaluate_perplexity(model, dataloader, device, tokenizer, pad_token_id=0):
    """
    Modified to compute TRUE BPC + Perplexity using CrossEntropyLoss 
    and actual character counting like your compute_bpc.
    """
    model.eval()
    total_loss = 0.0        # Sum of NLL (nats)
    total_chars = 0         # Total actual characters
    total_valid_tokens = 0  # For perplexity
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            
            # Shift labels (textbook causal LM)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = pad_token_id
            
            # Flatten for CE loss
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            labels_flat = labels.view(B * T)
            
            # Sum loss (ignores pads automatically)
            loss = loss_fn(logits_flat, labels_flat)
            total_loss += loss.item()
            
            # Count actual characters (BPC)
            for seq in input_ids:
                seq_no_pad = seq[seq != pad_token_id]
                text = tokenizer.decode(seq_no_pad.tolist())
                total_chars += len(text)
            
            # Count valid tokens (Perplexity)
            valid_mask = (labels_flat != pad_token_id)
            total_valid_tokens += valid_mask.sum().item()
    
    # Final metrics
    bpc = total_loss / (total_chars * math.log(2)) if total_chars > 0 else float('inf')
    perplexity = math.exp(total_loss / total_valid_tokens) if total_valid_tokens > 0 else float('inf')
    
    return {
        "bpc": bpc,
        "perplexity": perplexity,
        "total_chars": total_chars,
        "total_tokens": total_valid_tokens
    }


# def evaluate_perplexity(model, dataloader, device, pad_token_id=0):
#     """
#     Complete evaluation loop: computes mean BPC and perplexity over entire dataset.
    
#     Args:
#     - model: your trained causal LM
#     - dataloader: validation/test DataLoader 
#     - device: torch.device('cuda') or 'cpu'
#     - pad_token_id: your tokenizer.pad_token_id
    
#     Returns: dict with final metrics
#     """
#     model.eval()
#     total_nll = 0.0
#     total_tokens = 0
    
#     with torch.no_grad():
#         for batch in dataloader:
#             # Standard DataLoader batch format
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
            
#             # Forward pass
#             logits = model(input_ids, attention_mask)  # (B, S, V)
            
#             B, S, V = logits.shape
            
#             # Align for next-token prediction
#             shift_labels = input_ids[:, 1:]
#             logits_for_ppl = logits[:, :-1, :]
            
#             # Log probs for true tokens (natural log)
#             log_probs = F.log_softmax(logits_for_ppl, dim=-1)
#             gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
#             # Use attention_mask for precise padding (better than pad_token_id)
#             shift_mask = attention_mask[:, 1:]
#             nll_batch = -gathered_log_probs * shift_mask
#             valid_tokens_batch = shift_mask.sum().item()
            
#             # Accumulate global statistics
#             total_nll += nll_batch.sum().item()
#             total_tokens += valid_tokens_batch
    
#     # Final dataset-level metrics
#     avg_nll_per_token = total_nll / total_tokens if total_tokens > 0 else float('inf')
#     bpc = avg_nll_per_token / math.log(2)
#     perplexity = math.exp(avg_nll_per_token)
    
#     return {
#         "bpc": bpc,
#         "perplexity": perplexity,
#         "total_tokens": total_tokens
#     }


# def bits_per_character(batch_data, logits):
#     """
#     Handles DataLoader dict format: [{'input_ids': tensor, ...}, ...]
#     """
#     # Extract input_ids from dicts/list
#     if isinstance(batch_data, list) and len(batch_data) > 0:
#         if isinstance(batch_data[0], dict):
#             # DataLoader format: list of dicts
#             input_ids_list = [item['input_ids'] for item in batch_data]
#         else:
#             # List of tensors
#             input_ids_list = batch_data
#     else:
#         # Already a tensor
#         input_ids_list = batch_data
    
#     # Stack into batch tensor
#     input_ids = torch.stack(input_ids_list)
    
#     B, S, V = logits.shape
    
#     # Rest unchanged...
#     shift_labels = input_ids[:, 1:]
#     logits_for_ppl = logits[:, :-1, :]
    
#     log_probs = F.log_softmax(logits_for_ppl, dim=-1)
#     gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
#     pad_token_id = 0  # Set your pad token ID
#     mask = (shift_labels != pad_token_id).float()
    
#     nll = -gathered_log_probs * mask
#     total_nll = nll.sum()
#     total_tokens = mask.sum() + 1e-8
#     avg_nll_per_token = total_nll / total_tokens
    
#     bpc = avg_nll_per_token / math.log(2)
#     perplexity = torch.exp(avg_nll_per_token)
    
#     return {
#         "bpc": bpc.item(),
#         "perplexity": perplexity.item()
#     }



# def compute_perplexity(model, dataloader, device, pad_token_id=0):

#     model.eval()

#     total_loss = 0.0
#     total_tokens = 0

#     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')

#     with torch.no_grad():
#         for batch in dataloader:

#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)

#             logits = model(input_ids, attention_mask)

#             # 🔹 Create shifted labels
#             labels = input_ids.clone()
#             labels[:, :-1] = input_ids[:, 1:]
#             labels[:, -1] = pad_token_id

#             B, T, V = logits.shape

#             logits = logits.view(B * T, V)
#             labels = labels.view(B * T)

#             # 🔹 Sum loss over all tokens
#             loss = loss_fn(logits, labels)

#             total_loss += loss.item()

#             # 🔹 Count valid (non-pad) tokens
#             valid_tokens = (labels != pad_token_id).sum().item()
#             total_tokens += valid_tokens

#     # 🔹 Average loss per token
#     avg_loss = total_loss / total_tokens

#     # 🔹 Perplexity
#     perplexity = math.exp(avg_loss)

#     return perplexity

# import torch
# import torch.nn.functional as F

# import torch
# import torch.nn.functional as F
# import math

# def bits_per_character(input_ids, logits):
#     """
#     Calculate Bits Per Character (BPC) from input_ids and logits.
    
#     BPC = log2(perplexity) = (ln(perplexity)) / ln(2)
    
#     Returns: float (bits per character)
#     """
#     B, S, V = logits.shape
    
#     # Align for next-token prediction
#     shift_labels = input_ids[:, 1:]
#     logits_for_ppl = logits[:, :-1, :]
    
#     # Log probs for true next tokens (natural log)
#     log_probs = F.log_softmax(logits_for_ppl, dim=-1)  # ln(P)
#     gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
#     # Mask pads (adjust pad_token_id if needed)
#     pad_token_id = 0
#     mask = (shift_labels != pad_token_id).float()
    
#     # Average negative log likelihood (nats per token)
#     nll = -gathered_log_probs * mask
#     total_nll = nll.sum()
#     total_tokens = mask.sum() + 1e-8
#     avg_nll_per_token = total_nll / total_tokens  # L̄ (nats/token)
    
#     # Perplexity first (optional intermediate)
#     perplexity = torch.exp(avg_nll_per_token)
    
#     # BPC = log2(perplexity) = ln(perplexity)/ln(2)
#     bpc = avg_nll_per_token / math.log(2)  # Directly from nats to bits
    
#     return {
#         "bpc": bpc.item(),
#         "perplexity": perplexity.item(),
#         "nats_per_token": avg_nll_per_token.item()
#     }

# import torch
# import torch.nn.functional as F
# import math

# def bits_per_character(input_ids_list, logits):
#     """
#     Fixed: Handles list of tensors (batched padded inputs)
#     """
#     # Convert list → tensor if needed
#     if isinstance(input_ids_list, list):
#         input_ids = torch.stack(input_ids_list)  # List of (S,) → (B, S)
#     else:
#         input_ids = input_ids_list
    
#     B, S, V = logits.shape
    
#     # Align for next-token prediction
#     shift_labels = input_ids[:, 1:]
#     logits_for_ppl = logits[:, :-1, :]
    
#     # Log probs (natural log)
#     log_probs = F.log_softmax(logits_for_ppl, dim=-1)
#     gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
#     # Mask pads (YOUR pad token ID)
#     pad_token_id = 0  # CHANGE THIS to your tokenizer.pad_token_id
#     mask = (shift_labels != pad_token_id).float()
    
#     # Global average NLL (nats per token)
#     nll = -gathered_log_probs * mask
#     total_nll = nll.sum()
#     total_tokens = mask.sum() + 1e-8
#     avg_nll_per_token = total_nll / total_tokens
    
#     # BPC and PPL
#     bpc = avg_nll_per_token / math.log(2)
#     perplexity = torch.exp(avg_nll_per_token)
    
#     return {
#         "bpc": bpc.item(),
#         "perplexity": perplexity.item()
#     }

# def mean_perplexity(input_ids, logits):
#     """
#     Returns ONLY mean perplexity from input_ids and logits.
    
#     Args:
#     - input_ids: torch.LongTensor (B, S)
#     - logits: torch.FloatTensor (B, S, V)
    
#     Returns:
#     - float: mean perplexity across batch
#     """
#     B, S, V = logits.shape
    
#     # Align: logits[:-1] predict input_ids[1:]
#     shift_labels = input_ids[:, 1:]
#     logits_for_ppl = logits[:, :-1, :]
    
#     # Log probs for true next tokens
#     log_probs = F.log_softmax(logits_for_ppl, dim=-1)
#     gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
#     # Mask pads (assuming 0 is pad token)
#     pad_token_id = 0
#     mask = (shift_labels != pad_token_id).float()
    
#     # Mean negative log likelihood
#     nll = -gathered_log_probs * mask
#     token_counts = mask.sum(dim=-1) + 1e-8
#     mean_nll = nll.sum() / token_counts.sum()  # Global mean
    
#     return torch.exp(mean_nll).item()


# def compute_bpc(model, dataloader, device, pad_token_id=0):

#     model.eval()

#     total_loss = 0.0        # sum of NLL (in nats)
#     total_chars = 0         # total characters

#     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')

#     with torch.no_grad():
#         for batch in dataloader:

#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)

#             logits = model(input_ids, attention_mask)

#             # 🔹 Create shifted labels
#             labels = input_ids.clone()
#             labels[:, :-1] = input_ids[:, 1:]
#             labels[:, -1] = pad_token_id

#             B, T, V = logits.shape

#             logits = logits.view(B * T, V)
#             labels = labels.view(B * T)

#             # 🔹 Sum loss (not mean!)
#             loss = loss_fn(logits, labels)

#             total_loss += loss.item()

#             # 🔹 Count characters (IMPORTANT)
#             # You must store original text OR approximate from tokenizer
#             # If using BPE → sum lengths of decoded tokens
#             for seq in input_ids:
#                 # remove padding
#                 seq = seq[seq != pad_token_id]

#                 # approximate characters via decoding
#                 text = tokenizer.decode(seq.tolist())
#                 total_chars += len(text)

#     # 🔹 Final BPC
#     bpc = total_loss / (total_chars * math.log(2))

#     return bpc

def compute_loss(logits, input_ids, pad_id=0):
    
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id

    B, T, V = logits.shape
    logits = logits.contiguous().view(B * T, V)
    labels = labels.contiguous().view(B * T)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(logits, labels)
    return loss

def read_config(path: Path):
    
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

def encode_sentence(sentence):
    return tokenizer.encode(sentence)

# Reusing parta.check methods # Optimize later, something around threads
def encode_corpus(filepath):
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())

    print(f"Loaded {len(corpus)} sentences.")
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel encoding.")
    
    print("Encoding corpus in parallel...")
    with Pool(processes=num_processes) as pool:
        encode_func = partial(encode_sentence)
        encoded_corpus = list(tqdm(
            pool.imap(encode_func, corpus, chunksize=max(1, len(corpus) // (num_processes * 4))),
            total=len(corpus),
            desc="Encoding"
        ))

    return encoded_corpus



def main(args):
    # raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")
    # random.seed(100)

    config = read_config(Path("./parta/data/case2/config.json"))
    # breakpoint()

    global device
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"

    print(f"Device: {device}")

    t1 = time.time()
    global tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    t2 = time.time()
    print(f"Tokenizer loaded from {args.tokenizer_path} in {t2-t1} seconds.")

    T1 = time.time()
    t1 = time.time()
    train_tokens_list = encode_corpus(args.train_path)
    valid_tokens_list = encode_corpus(args.valid_path)
    t2 = time.time()
    print(f"Training and validation tokens generated in {t2-t1} seconds.")

    t1 = time.time()
    train_tensor = [torch.as_tensor(seq, dtype=torch.long) for seq in train_tokens_list]
    valid_tensor = [torch.as_tensor(seq, dtype=torch.long) for seq in valid_tokens_list]
    train_attention_masks  = [torch.ones(len(seq), dtype=torch.int64) for seq in train_tokens_list]
    valid_attention_masks  = [torch.ones(len(seq), dtype=torch.int64) for seq in valid_tokens_list]
    t2 = time.time()
    print(f"Tensor list convertion in {t2-t1} seconds.")
    
    t1 = time.time()
    # train_dataset = CustomDataset(train_tensor, train_attention_masks)
    # valid_dataset = CustomDataset(valid_tensor, valid_attention_masks)
    # train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn_2, num_workers=cpu_count())
    # valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn_2, num_workers=cpu_count())
    
    batch_size = 128
    train_batches = []
    train_batches_padded = []
    for start in range(0, len(train_tensor), batch_size):
        end = start + batch_size
        batch = {}
        batch['input_ids'] = train_tensor[start : end]
        batch['attention_mask'] = train_attention_masks[start : end]
        train_batches.append(batch)
        padded_batch = collate_fn(batch)
        padded_batch = {k: v.to(device) for k, v in padded_batch.items()}
        train_batches_padded.append(padded_batch)
    valid_batches = []    
    valid_batches_padded = []
    for start in range(0, len(valid_tensor), batch_size):
        end = start + batch_size
        batch = {}
        batch['input_ids'] = valid_tensor[start : end]
        batch['attention_mask'] = valid_attention_masks[start : end]
        valid_batches.append(batch)
        padded_batch = collate_fn(batch)
        padded_batch = {k: v.to(device) for k, v in padded_batch.items()}
        valid_batches_padded.append(padded_batch)

    t2 = time.time()
    print(f"Data loaded in {t2-t1} seconds.")
    T2 = time.time()
    print(f"Data pre-processed in {T2-T1} seconds")

    model = LanguageModel(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    epochs = 1000

    best_val_loss = float('inf')
    patience = 3

    os.makedirs(args.output_model_path, exist_ok=True)
    pad_id = tokenizer.get_pad_id()

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for i in range(len(train_batches_padded)):
            optimizer.zero_grad()
            logits = model(train_batches_padded[i]['input_ids'], train_batches_padded[i]['attention_mask'])
            # breakpoint()
            loss = compute_loss(logits, train_batches_padded[i]['input_ids'], pad_id)
            # breakpoint()
            loss.backward()
            optimizer.step()
            train_loss += loss
        avg_train_loss = train_loss/len(train_batches_padded)
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}")

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for i in range(len(valid_batches_padded)):
                logits = model(valid_batches_padded[i]['input_ids'], valid_batches_padded[i]['attention_mask'])
                loss = compute_loss(logits, valid_batches_padded[i]['input_ids'], pad_id)
                val_loss += loss
        avg_val_loss = val_loss/len(valid_batches_padded)
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_model_path, f"model.pt"))
        else:
            patience_counter += 1


        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, patience Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # val_bpc = compute_bpc(model, valid_batches_padded, device)
    # print(f"Validation BPC: {val_bpc}")

    # val_ppl = compute_perplexity(model, valid_batches_padded, device)
    # print(f"Validation Perplexity: {val_ppl:.4f}")
    model.load_state_dict(torch.load(os.path.join(args.output_model_path, "model.pt")))
    model.eval()
    model.to(device)
    # Usage in your training script (replace line 421):
    metrics = evaluate_perplexity(model, valid_batches_padded, device, tokenizer=tokenizer, pad_token_id=tokenizer.get_pad_id())
    print(f"Validation BPC: {metrics['bpc']:.4f} bits/char")
    print(f"Validation PPL: {metrics['perplexity']:.4f}")
    print(f"Chars: {metrics['total_chars']:,}, Tokens: {metrics['total_tokens']:,}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
