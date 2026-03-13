class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        # 1. Initialize vocab with base characters -> Start with individual characters or bytes as initial vocab
        self.vocab_size = vocab_size
        self.tokens_to_integers = {}
        self.integers_to_tokens = {}

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
        unique_chars = set(char_list)

        # Parallelize loop later
        for index, char in enumerate(unique_chars):
            self.tokens_to_integers[char] = index + 4
            self.integers_to_tokens[index + 4] = char

        # # Test if initial vocab works
        # decoded = self.encode(corpus[0])
        # encoded = self.decode(decoded)
        # for sentence in corpus:
        #     decoded = self.encode(sentence)
        #     encoded = self.decode(decoded)
        #     if sentence != encoded:
        #         print("check")
        #         breakpoint()

        # breakpoint()
        #5. Think about merge operation.
            
    def encode(self, text):
        token_ids = [None]*len(text)
        for index, char in enumerate(list(text)):
            token_ids[index] = self.tokens_to_integers[char]
        return token_ids

    def decode(self, token_ids):
        text_list = [None] * len(token_ids)
        for index, token_index in enumerate(token_ids):
            text_list[index] = self.integers_to_tokens[token_index]
        text = "".join(text_list)
        return text

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
    
    def get_vocab_size(self):
        # raise NotImplementedError("Get vocab size method not implemented yet.")
        return self.vocab_size
    
    def get_pad_id(self):
        # raise NotImplementedError("Get unk id method not implemented yet.")
        return int(0)


    def get_unk_id(self):
        # raise NotImplementedError("Get unk id method not implemented yet.")
        return int(1)
    
    def get_sos_id(self):
        return int(2)
        # raise NotImplementedError("Get unk id method not implemented yet.")

    def get_eos_id(self):
        return int(3)
        # raise NotImplementedError("Get unk id method not implemented yet.")
