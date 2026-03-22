# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn
from .utils import dummy_function 

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
from torch.optim.lr_scheduler import LambdaLR

device = None
tokenizer = None

def evaluate_bpc_2(model, data_loader, tokenizer, device, pad_id=0):
    """
    Evaluates the model and calculates Bits Per Character (BPC).
    """
    model.eval()
    
    # Using reduction='sum' exactly calculates the numerator: Sum(L_i * T_i)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    
    total_loss_sum = 0.0
    total_characters = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            x = input_ids[:, :-1]
            mask = attention_mask[:, :-1]
            targets = input_ids[:, 1:]
            
            logits = model(input_ids=x, attention_mask=mask)
            
            # 1. Calculate Numerator: Sum of all token losses in the batch
            loss_sum = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss_sum += loss_sum.item()
            
            # 2. Calculate Denominator: Total characters in the target sequences
            # We loop through the batch to decode the targets back into strings
            for i in range(targets.size(0)):
                # Extract only the valid (non-padded) tokens for this specific sequence
                valid_tokens = targets[i][targets[i] != pad_id].tolist()
                
                # Use your Part B tokenizer to turn the tokens back into a string
                decoded_str = tokenizer.decode(valid_tokens)
                
                # Count the actual characters
                total_characters += len(decoded_str)

    # 3. Apply the final formula
    # BPC = sum(losses) / (Total_Chars * ln(2))
    bpc = total_loss_sum / (total_characters * math.log(2))
    
    return bpc

def get_cosine_scheduler(optimizer, num_training_steps, warmup_steps):
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        
        progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def get_optimizer(model, lr=3e-4, weight_decay=0.01):
    decay = set()
    no_decay = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower():
            no_decay.add(name)
        else:
            decay.add(name)

    param_dict = {name: param for name, param in model.named_parameters()}

    optim_groups = [
        {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    return optimizer



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
            for seq in labels:
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

    config = read_config(Path("./parta/data/case1/config.json"))
    # breakpoint()

    global device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")

    t1 = time.time()
    global tokenizer
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
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
    
    batch_size = 32
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

    epochs = 1000
    total_steps = epochs * len(train_batches_padded)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # model = LanguageModelEncoder(config=config).to(device)
    model = LanguageModel(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = get_optimizer(model, lr=5e-3, weight_decay=1e-4)
    # scheduler = get_cosine_scheduler(optimizer, total_steps, warmup_steps)
    

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
            # scheduler.step() 
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
    model = LanguageModel(config=config).to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_model_path, "model.pt")))
    model.eval()
    model.to(device)
    # Usage in your training script (replace line 421):
    metrics = evaluate_perplexity(model, valid_batches_padded, device, tokenizer=tokenizer, pad_token_id=tokenizer.get_pad_id())
    print(f"Validation BPC: {metrics['bpc']:.4f} bits/char")
    print(f"Validation PPL: {metrics['perplexity']:.4f}")
    print(f"Chars: {metrics['total_chars']:,}, Tokens: {metrics['total_tokens']:,}")
    m2 = evaluate_bpc_2(model, valid_batches_padded, tokenizer, device, tokenizer.get_pad_id())
    print(f"Other bpc {m2}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
