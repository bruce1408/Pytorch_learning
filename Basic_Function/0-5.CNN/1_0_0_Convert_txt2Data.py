# Defined in Section 4.6.1
import torch
from collections import defaultdict, Counter
from nltk.corpus import sentence_polarity


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write("\n".join(vocab.idx_to_token))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return Vocab(tokens)


def load_sentence_polarity():
    vocab = Vocab.build(sentence_polarity.sents())

    # train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
    #               for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
    #              + [(vocab.convert_tokens_to_ids(sentence), 1)
    #                 for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    train_data_pos = list()
    for sentence in sentence_polarity.sents(categories='pos')[:4000]:
        train_data_pos.append((vocab.convert_tokens_to_ids(sentence), 0))

    train_data_neg = list()
    for sentence in sentence_polarity.sents(categories='neg')[:4000]:
        train_data_neg.append((vocab.convert_tokens_to_ids(sentence), 1))

    train_data_pos.extend(train_data_neg)


    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
                + [(vocab.convert_tokens_to_ids(sentence), 1)
                   for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data_pos, test_data, vocab


if __name__ == "__main__":
    train_data, _, _ = load_sentence_polarity()
    print(train_data.__len__())
    print(train_data)
