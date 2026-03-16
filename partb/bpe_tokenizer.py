import json
import os
import re
import time

min_freq = 5
timeout = 10500

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        # 1. Initialize vocab with base characters -> Start with individual characters or bytes as initial vocab
        self.vocab_size = vocab_size
        self.tokens_to_integers = {}
        self.integers_to_tokens = {}
        self.merges = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    def train(self, corpus):
        # 1. Initialize vocabulary
        time_start = time.time()
        self.tokens_to_integers["<PAD>"] = self.get_pad_id()
        self.integers_to_tokens[self.get_pad_id()] = "<PAD>"
        self.tokens_to_integers["<UNK>"] = self.get_unk_id()
        self.integers_to_tokens[self.get_unk_id()] = "<UNK>"
        self.tokens_to_integers["<SOS>"] = self.get_sos_id()
        self.integers_to_tokens[self.get_sos_id()] = "<SOS>"
        self.tokens_to_integers["<EOS>"] = self.get_eos_id()
        self.integers_to_tokens[self.get_eos_id()] = "<EOS>"
        for special_token in self.special_tokens:
            if special_token not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]:
                self.tokens_to_integers[special_token] = len(self.tokens_to_integers)
                self.integers_to_tokens[len(self.tokens_to_integers)] = special_token
        

        #2. Create a character level vocab, extracting unique characters from training corpus.
        # Add code to handle unknown variables later, like 
        # english alphabets
        # Hindi unicode: U+0900 to U+097F
        # Numbers in hindi and english

        flag = True

        word_freq = {}
        words_spaced = []
        for sentence in corpus:
            sentence_words = re.findall(r'\S+|\s+', sentence)
            for word in sentence_words: # not parallel
                if word.isspace():
                    words_spaced[-1] += word
                else:
                    words_spaced.append(word)
                
            
            
        for word in words_spaced:
            word_freq[word] = word_freq.get(word, 0) + 1

        
        char_list = []
        for word in word_freq.keys():
            char_list.extend(list(word))
            # breakpoint()
        self.vocab = set(char_list)


        # Parallelize loop later
        offset = len(self.tokens_to_integers)
        for index, char in enumerate(self.vocab):
            self.tokens_to_integers[char] = index + offset
            self.integers_to_tokens[index + offset] = char

        # convert to tokens
        word_tokens_freq = {}
        for word, freq in word_freq.items():
            word_tokens_freq[tuple(self.initial_tokenize(word))] = freq

        print(f"Vocab_size: {len(self.vocab)}")

        while len(self.vocab) < self.vocab_size:
            # Build pairs, optimize for time later:
            
            pairs = {}
            for word_tokens, freq in word_tokens_freq.items(): # should not make parallel, else race condition
                for i in range(len(word_tokens)-1):
                    pair = (word_tokens[i], word_tokens[i+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
                    # update only merged pairs, rather than building pairs from scratch

            if not pairs:
                print("No more pairs available for merging, so combining remaining word tokens.")
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_freq:
                print(f"Breaking out of loop as best pair frequency is less than min_freq for vocab size:{len(self.vocab)}.")
                flag = False
                break
            self.merges.append(best_pair)

            best_pair_token = self.integers_to_tokens[best_pair[0]] + self.integers_to_tokens[best_pair[1]]
            self.vocab.add(best_pair_token)
            best_pair_id = len(self.tokens_to_integers)
            self.tokens_to_integers[best_pair_token] = best_pair_id
            self.integers_to_tokens[best_pair_id] = best_pair_token

            # Update tokens
            merged_word_tokens_freq = {}
            for word_tokens, freq in word_tokens_freq.items():
                word_tokens_list = list(word_tokens)
                i=0
                while i < len(word_tokens_list)-1:
                    pair = (word_tokens_list[i], word_tokens_list[i+1])
                    if pair == best_pair:
                        word_tokens_list[i] = best_pair_id
                        word_tokens_list.pop(i+1)
                        # track merged token pairs
                    else:
                        i += 1
                merged_word_tokens = tuple(word_tokens_list)       
                # merged_word_tokens_freq[merged_word_tokens] = freq
                merged_word_tokens_freq[merged_word_tokens] = merged_word_tokens_freq.get(merged_word_tokens, 0) + freq
            word_tokens_freq = merged_word_tokens_freq
                

            print(f"Vocab_size: {len(self.vocab)}")
            if time.time() - time_start > timeout:
                print(f"Breaking out of loop as training time exceeded {timeout} seconds for vocab size:{len(self.vocab)}.")
                break
            # breakpoint()


        if len(self.vocab) < self.vocab_size and flag:

            word_tokens_freq = {}
            words_spaced = []
            sentence_word_list = []
            # parallel
            for sentence in corpus:
                sentence_words = re.findall(r'\S+|\s+', sentence)
                for word in sentence_words: # not parallel
                    if word.isspace():
                        words_spaced[-1] += word
                    else:
                        words_spaced.append(word)
                sentence_word_list.append(words_spaced)

            sentence_token_list = []        
            for sentence_words in sentence_word_list:
                word_list = []
                for word in sentence_words:
                    word_list.append(self.tokens_to_integers[word])
                sentence_token_list.append(word_list)

            for word_list in sentence_token_list:
                word_tokens_freq[tuple(word_list)] = word_tokens_freq.get(tuple(word_list),0) + 1

            while len(self.vocab) < self.vocab_size:
                # Build pairs, optimize for time later:
                
                pairs = {}
                for word_tokens, freq in word_tokens_freq.items(): # should not make parallel, else race condition
                    for i in range(len(word_tokens)-1):
                        pair = (word_tokens[i], word_tokens[i+1])
                        pairs[pair] = pairs.get(pair, 0) + freq
                        # update only merged pairs, rather than building pairs from scratch

                if not pairs:
                    print("No more pairs available for merging.")
                    break

                best_pair = max(pairs, key=pairs.get)
                if pairs[best_pair] < min_freq:
                    print(f"Breaking out of loop as best pair frequency is less than min_freq for vocab size:{len(self.vocab)}.")
                    break
                self.merges.append(best_pair)

                best_pair_token = self.integers_to_tokens[best_pair[0]] + self.integers_to_tokens[best_pair[1]]
                self.vocab.add(best_pair_token)
                best_pair_id = len(self.tokens_to_integers)
                self.tokens_to_integers[best_pair_token] = best_pair_id
                self.integers_to_tokens[best_pair_id] = best_pair_token

                # Update tokens
                merged_word_tokens_freq = {}
                for word_tokens, freq in word_tokens_freq.items():
                    word_tokens_list = list(word_tokens)
                    i=0
                    while i < len(word_tokens_list)-1:
                        pair = (word_tokens_list[i], word_tokens_list[i+1])
                        if pair == best_pair:
                            word_tokens_list[i] = best_pair_id
                            word_tokens_list.pop(i+1)
                            # track merged token pairs
                        else:
                            i += 1
                    merged_word_tokens = tuple(word_tokens_list)       
                    # merged_word_tokens_freq[merged_word_tokens] = freq
                    merged_word_tokens_freq[merged_word_tokens] = merged_word_tokens_freq.get(merged_word_tokens, 0) + freq
                word_tokens_freq = merged_word_tokens_freq
                    

                print(f"Vocab_size: {len(self.vocab)}")
                if time.time() - time_start > timeout:
                    print(f"Breaking out of loop as training time exceeded {timeout} seconds for vocab size:{len(self.vocab)}.")
                    break


    def initial_tokenize(self, text):
        token_ids = [None]*len(text)
        for index, char in enumerate(list(text)):
            token_ids[index] = self.tokens_to_integers.get(char, self.get_unk_id())
        return token_ids

            
    def encode(self, text, add_sos_eos=False):

        # token_ids = [None]*len(text)
        # for index, char in enumerate(list(text)):
        #     token_ids[index] = self.tokens_to_integers.get(char, self.get_unk_id())
        token_ids = self.initial_tokenize(text)
        
        
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
        if add_sos_eos:
            token_ids = [self.get_sos_id()] + token_ids + [self.get_eos_id()]

        return token_ids

    def decode(self, token_ids):

        for merge in self.merges:
            i=0
            while i < len(token_ids)-1:
                # breakpoint()
                # if token_ids in [self.get_pad_id(), self.get_unk_id(), self.get_sos_id(), self.get_eos_id()]:
                #     i += 1
                if merge == (token_ids[i], token_ids[i+1]):
                    merged_token = self.integers_to_tokens[merge[0]] + self.integers_to_tokens[merge[1]]
                    merged_id = self.tokens_to_integers[merged_token]
                    # breakpoint()
                    token_ids[i] = merged_id
                    token_ids.pop(i+1)
                else:
                    i += 1
        # breakpoint()
        # text_list = [None] * len(token_ids)
        text_list = []
        for index, token_index in enumerate(token_ids):
            # breakpoint()
            # text_list[index] = self.integers_to_tokens[str(token_index)]
            if token_index in [self.get_pad_id(), self.get_unk_id(), self.get_sos_id(), self.get_eos_id()]:
                continue
            text_list.append(self.integers_to_tokens[token_index])
        # breakpoint()
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
