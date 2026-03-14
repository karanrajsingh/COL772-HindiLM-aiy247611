import json
import os

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        # 1. Initialize vocab with base characters -> Start with individual characters or bytes as initial vocab
        self.vocab_size = vocab_size
        self.tokens_to_integers = {}
        self.integers_to_tokens = {}
        self.merges = []

    def train(self, corpus):
        # 1. Initialize vocabulary
        self.tokens_to_integers["<PAD>"] = self.get_pad_id()
        self.integers_to_tokens[self.get_pad_id()] = "<PAD>"
        self.tokens_to_integers["<UNK>"] = self.get_unk_id()
        self.integers_to_tokens[self.get_unk_id()] = "<UNK>"
        self.tokens_to_integers["<SOS>"] = self.get_sos_id()
        self.integers_to_tokens[self.get_sos_id()] = "<SOS>"
        self.tokens_to_integers["<EOS>"] = self.get_eos_id()
        self.integers_to_tokens[self.get_eos_id()] = "<EOS>"
        #2. Create a character level vocab, extracting unique characters from training corpus.
        # Add code to handle unknown variables later, like 
        # english alphabets
        # Hindi unicode: U+0900 to U+097F
        # Numbers in hindi and english
        char_list = []
        # Can this be efficient?
        for sentence in corpus:
            char_list.extend(list(sentence))
        self.vocab = set(char_list)
        # corpus_str = "".join(char_list)

        # Parallelize loop later
        for index, char in enumerate(self.vocab):
            self.tokens_to_integers[char] = index + 4
            self.integers_to_tokens[index + 4] = char

        # convert to tokens
        corpus_tokens = []
        for sentence in corpus:
            corpus_tokens.append(self.encode(sentence))

        print(f"Vocab_size: {len(self.vocab)}")

        while len(self.vocab) < self.vocab_size:
            # Build pairs, optimize for time later:
            
            pairs = {}
            for sentence_tokens in corpus_tokens:
                for i in range(len(sentence_tokens)-1):
                    # pair = self.integers_to_tokens[sentence_tokens[i]] + self.integers_to_tokens[sentence_tokens[i+1]]
                    pair = (sentence_tokens[i], sentence_tokens[i+1])
                    pairs.setdefault(pair, 0)
                    # if pairs.get(pair) is None:
                    #     pairs[pair] = 0

        # Update pair counts -> Don't make Parallel, else race condition
            for sentence_tokens in corpus_tokens:
                for i in range(len(sentence_tokens)-1):
                    # pair = self.integers_to_tokens[sentence_tokens[i]] + self.integers_to_tokens[sentence_tokens[i+1]]
                    pair = (sentence_tokens[i], sentence_tokens[i+1])
                    pairs[pair] += 1


            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            best_pair_token = self.integers_to_tokens[best_pair[0]] + self.integers_to_tokens[best_pair[1]]
            self.vocab.add(best_pair_token)
            best_pair_id = len(self.tokens_to_integers)
            self.tokens_to_integers[best_pair_token] = best_pair_id
            self.integers_to_tokens[best_pair_id] = best_pair_token

            # Update tokens
            for sentence_tokens in corpus_tokens:
                i=0
                while i < len(sentence_tokens)-1:
                    # pair = self.integers_to_tokens[sentence_tokens[i]] + self.integers_to_tokens[sentence_tokens[i+1]]
                    pair = (sentence_tokens[i], sentence_tokens[i+1])
                    if pair == best_pair:
                        sentence_tokens[i] = best_pair_id
                        sentence_tokens.pop(i+1)
                    else:
                        i += 1

            print(f"Vocab_size: {len(self.vocab)}")
            # breakpoint()

        #5. Think about merge operation.
        
        # while len(self.vocab) < self.vocab_size-1:
        #     for sentence in corpus:
        #         tokens = list(sentence)
        #         for i in range(len(tokens) - 1):
        #             pair = tokens[i] + tokens[i+1]
        #             breakpoint()


            
    def encode(self, text):

        token_ids = [None]*len(text)
        for index, char in enumerate(list(text)):
            token_ids[index] = self.tokens_to_integers[char]
        
        
        for merge in self.merges:
            i=0
            while i < len(token_ids)-1:
                if merge == (token_ids[i], token_ids[i+1]):
                    # breakpoint()
                    merged_token = self.integers_to_tokens[merge[0]] + self.integers_to_tokens[merge[1]]
                    merged_id = self.tokens_to_integers[merged_token]
                    token_ids[i] = merged_id
                    token_ids.pop(i+1)
                else:
                    i += 1
        return token_ids

    def decode(self, token_ids):

        for merge in self.merges:
            i=0
            while i < len(token_ids)-1:
                # breakpoint()
                if merge == (token_ids[i], token_ids[i+1]):
                    merged_token = self.integers_to_tokens[merge[0]] + self.integers_to_tokens[merge[1]]
                    merged_id = self.tokens_to_integers[merged_token]
                    breakpoint()
                    token_ids[i] = merged_id
                    token_ids.pop(i+1)
                else:
                    i += 1
        # breakpoint()
        text_list = [None] * len(token_ids)
        for index, token_index in enumerate(token_ids):
            # breakpoint()
            # text_list[index] = self.integers_to_tokens[str(token_index)]
            text_list[index] = self.integers_to_tokens[token_index]
        text = "".join(text_list)
        return text

    def save(self, filepath):
        file = os.path.join(filepath, "my_bpe.json")
        tokenizer_dict = {}
        tokenizer_dict["tokens_to_integers"] = self.tokens_to_integers
        tokenizer_dict["integers_to_tokens"] = self.integers_to_tokens
        tokenizer_dict["merges"] = self.merges
        with open(file, "w") as f:
            json.dump(tokenizer_dict, f, indent=2)
        # raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        # raise NotImplementedError("Load method not implemented yet.")
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_dict = json.load(f)
        self.tokens_to_integers = tokenizer_dict["tokens_to_integers"]
        for key, value in tokenizer_dict["integers_to_tokens"].items():
            self.integers_to_tokens[int(key)] = value
        # self.integers_to_tokens = tokenizer_dict["integers_to_tokens"]
        for pair in tokenizer_dict["merges"]:
            self.merges.append(tuple(pair))
        # self.merges = tokenizer_dict["merges"]

    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_pad_id(self):
        return int(0)


    def get_unk_id(self):
        return int(1)
    
    def get_sos_id(self):
        return int(2)

    def get_eos_id(self):
        return int(3)
