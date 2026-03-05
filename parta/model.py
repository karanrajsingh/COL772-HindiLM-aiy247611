import torch
import torch.nn as nn
from typing import Any, Dict, List
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


max_sequence_len = 512
device = None

def set_device():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

def get_positional_embeddings(d_model):
    global device, max_sequence_len
    set_device()
    pos_num = torch.arange(start=1, end=max_sequence_len+1, dtype=torch.float32, device=device).unsqueeze(dim=1).expand(max_sequence_len, d_model)
    pos_den = torch.arange(start=1, end=d_model+1, dtype=torch.float32, device=device).unsqueeze(dim=0).expand(max_sequence_len, d_model)
    pos_den = torch.pow(10000,(2*torch.floor((pos_den+1)/2))/d_model)
    pos_val = pos_num/pos_den
    sin_mask = (torch.arange(start=1, end=d_model+1, dtype=torch.float32, device=device)%2 == float(1)).unsqueeze(dim=0).expand(max_sequence_len, d_model)
    cos_mask = (torch.arange(start=1, end=d_model+1, dtype=torch.float32, device=device)%2 == float(0)).unsqueeze(dim=0).expand(max_sequence_len, d_model)
    pos_embed = torch.zeros(max_sequence_len, d_model, dtype=torch.float32, device=device)
    pos_embed[sin_mask] = torch.sin(pos_val[sin_mask])
    pos_embed[cos_mask] = torch.cos(pos_val[cos_mask])
    return pos_embed
    

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()
        # 1. Input embeddings
        self.embed = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        # 2. Positional Encoding
        self.pos_embed = get_positional_embeddings(self.config['d_model'])
        # 3. Transformer block
        self.layer_norm = nn.ModuleDict()       
        self.query = nn.ModuleDict()
        self.key = nn.ModuleDict()
        self.value = nn.ModuleDict()
        self.output = nn.ModuleDict()
        self.up_proj = nn.ModuleDict()
        self.down_proj = nn.ModuleDict()
        for layer in range(1, self.config['n_layers']+1):
            # a. LayerNorm(X(l-1))
            self.layer_norm[f'{layer}_1'] = nn.LayerNorm(self.config['d_model'], elementwise_affine=True, bias=True)
            # b. Multi-Head Attention
            for head in range(1, self.config['n_heads']+1):
                self.query[f'{layer}_{head}'] = nn.Linear(self.config['d_model'], self.config['d_head'], bias=False)
                self.key[f'{layer}_{head}'] = nn.Linear(self.config['d_model'], self.config['d_head'], bias=False)
                self.value[f'{layer}_{head}'] = nn.Linear(self.config['d_model'], self.config['d_head'], bias=False)
            self.output[f'{layer}'] = nn.Linear(self.config['d_model'], self.config['d_model'], bias=False)
            # c. Residual Connection: Xnet(l-1) = X(l-1) + Z1(l)
            # Handle in forward
            # d. LayerNorm(Xnet(l-1))
            self.layer_norm[f'{layer}_2'] = nn.LayerNorm(self.config['d_model'], elementwise_affine=True, bias=True)
            # e. UpProjection
            self.up_proj[f'{layer}'] = nn.Linear(self.config['d_model'], 4*self.config['d_model'], bias=True)
            # f. Non-linearity
            self.activation = nn.GELU()
            # g. DownProjection
            self.down_proj[f'{layer}'] = nn.Linear(4*self.config['d_model'], self.config['d_model'], bias=True)
            # h. Residual Connection: X(l) = Xnet(l-1) + Z2(l)
            # Handle in forward
        # 4. Final Normalization
        self.layer_norm['final'] = nn.LayerNorm(self.config['d_model'], elementwise_affine=True, bias=True)
        # 5. Logits
        self.logits = nn.Linear(self.config['d_model'], self.config['vocab_size'], bias=False)


    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        # raise NotImplementedError("Implement set_weights as described in assignment document")
        # As per assignment, no transpose where pre-multiplication is done, and transpose where post-multiplication is done.
        with torch.no_grad():
            state_dict = {}
            state_dict['embed.weight'] = weights['W_vocab'].t()
            for layer in range(1, self.config['n_layers']+1):
                state_dict[f'layer_norm.{layer}_1.weight'] = weights[f'gamma_{layer}_1']
                state_dict[f'layer_norm.{layer}_1.bias'] = weights[f'beta_{layer}_1']
                for head in range(1, self.config['n_heads']+1):
                    state_dict[f'query.{layer}_{head}.weight'] = weights[f'W_{layer}_Q_{head}']
                    state_dict[f'key.{layer}_{head}.weight'] = weights[f'W_{layer}_K_{head}']
                    state_dict[f'value.{layer}_{head}.weight'] = weights[f'W_{layer}_V_{head}']
                state_dict[f'output.{layer}.weight'] = weights[f'W_{layer}_O']
                state_dict[f'layer_norm.{layer}_2.weight'] = weights[f'gamma_{layer}_2']
                state_dict[f'layer_norm.{layer}_2.bias'] = weights[f'beta_{layer}_2']
                state_dict[f'up_proj.{layer}.weight'] = weights[f'W_{layer}_up'].t()
                state_dict[f'up_proj.{layer}.bias'] = weights[f'b_{layer}_up']
                state_dict[f'down_proj.{layer}.weight'] = weights[f'W_{layer}_down'].t()
                state_dict[f'down_proj.{layer}.bias'] = weights[f'b_{layer}_down']
            state_dict['layer_norm.final.weight'] = weights['gamma_final']  
            state_dict['layer_norm.final.bias'] = weights['beta_final']
            state_dict['logits.weight'] = weights['W_devocab'].t()
            self.load_state_dict(state_dict)



    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        breakpoint()
        raise NotImplementedError("Implement forward as described in assignment document")

        



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)
    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID

    global device, max_sequence_len
    input_list = batch['input_ids']
    mask_list = batch['attention_mask']
    pad_len = max_sequence_len - input_list[0].size()[0]
    input_list[0] = F.pad(input_list[0], (0, pad_len), value=PAD_ID)
    mask_list[0] = F.pad(mask_list[0], (0, pad_len), value=0)
    padded_input = pad_sequence(input_list, batch_first=True, padding_value=PAD_ID)
    padded_mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
    padded_batch = {}
    padded_batch['input_ids'] = padded_input
    padded_batch['attention_mask'] = padded_mask
    return padded_batch

    # raise NotImplementedError("Implement collate_fn as described in assignment document")