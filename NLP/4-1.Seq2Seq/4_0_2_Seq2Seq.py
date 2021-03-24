# coding: utf-8
import os
import math
import random
import time
import spacy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

"""
paper: Sequence to Sequence Learning with Neural Networks
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(torch.__version__)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def tokenize_de(text):
    """
    每个字符倒序
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    """
    切词且正常顺序
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


train_data, valid_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(SRC, TRG))
# We can also print out an example, making sure the source sentence is reversed:

print(vars(train_data.examples[0]))
# The period is at the beginning of the German (src) sentence, so it looks like the sentence has been correctly
# reversed.

# Next, we'll build the *vocabulary* for the source and target languages. The vocabulary is used to associate each
# unique token with an index (an integer). The vocabularies of the source and target languages are distinct.
#
# Using the `min_freq` argument, we only allow tokens that appear at least 2 times to appear in our vocabulary.
# Tokens that appear only once are converted into an `<unk>` (unknown) token.
#
# It is important to note that our vocabulary should only be built from the training set and not the validation/test
# set. This prevents "information leakage" into our model, giving us artifically inflated validation/test scores.


SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

# The final step of preparing the Dataset is to create the iterators. These can be iterated on to return a batch of Dataset
# which will have a `src` attribute (the PyTorch tensors containing a batch of numericalized source sentences) and a
# `trg` attribute (the PyTorch tensors containing a batch of numericalized target sentences). Numericalized is just a
# fancy way of saying they have been converted from a sequence of readable tokens to a sequence of corresponding
# indexes, using the vocabulary.
#
# We also need to define a `torch.device`. This is used to tell TorchText to put the tensors on the GPU or not. We
# use the `torch.cuda.is_available()` function, which will return `True` if a GPU is detected on our computer. We
# pass this `device` to the iterator.
#
# When we get a batch of examples using an iterator we need to make sure that all of the source sentences are padded
# to the same length, the same with the target sentences. Luckily, TorchText iterators handle this for us!
#
# We use a `BucketIterator` instead of the standard `Iterator` as it creates batches in such a way that it minimizes
# the amount of padding in both the source and target sentences.


print(f"Number of training examples: {len(train_data.examples)}")  # 29000
print(f"Number of validation examples: {len(valid_data.examples)}")  # 1014
print(f"Number of testing examples: {len(test_data.examples)}")  # 1000


print(vars(train_data.examples[0]))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


# ## Building the Seq2Seq Model
#
# We'll be building our model in three parts. The encoder, the decoder and a seq2seq model that encapsulates the
# encoder and decoder and will provide a way to interface with each.
#
# ### Encoder
#
# First, the encoder, a 2 layer LSTM. The paper we are implementing uses a 4-layer LSTM, but in the interest of
# training time we cut this down to 2-layers. The concept of multi-layer RNNs is easy to expand from 2 to 4 layers.
#
# For a multi-layer RNN, the input sentence, $X$, after being embedded goes into the first (bottom) layer of the RNN
# and hidden states, $H=\{h_1, h_2, ..., h_T\}$, output by this layer are used as inputs to the RNN in the layer
# above. Thus, representing each layer with a superscript, the hidden states in the first layer are given by:
#
# $$h_t^1 = \text{EncoderRNN}^1(e(x_t), h_{t-1}^1)$$
#
# The hidden states in the second layer are given by:
#
# $$h_t^2 = \text{EncoderRNN}^2(h_t^1, h_{t-1}^2)$$
#
# Using a multi-layer RNN also means we'll also need an initial hidden state as input per layer, $h_0^l$, and we will
# also output a context vector per layer, $z^l$.
#
# Without going into too much detail about LSTMs (see [this](
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/) blog post to learn more about them), all we need to
# know is that they're a type of RNN which instead of just taking in a hidden state and returning a new hidden state
# per time-step, also take in and return a *cell state*, $c_t$, per time-step.
#
# $$\begin{align*}
# h_t &= \text{RNN}(e(x_t), h_{t-1})\\
# (h_t, c_t) &= \text{LSTM}(e(x_t), h_{t-1}, c_{t-1})
# \end{align*}$$
#
# We can just think of $c_t$ as another type of hidden state. Similar to $h_0^l$, $c_0^l$ will be initialized to a
# tensor of all zeros. Also, our context vector will now be both the final hidden state and the final cell state,
# i.e. $z^l = (h_T^l, c_T^l)$.
#
# Extending our multi-layer equations to LSTMs, we get:
#
# $$\begin{align*}
# (h_t^1, c_t^1) &= \text{EncoderLSTM}^1(e(x_t), (h_{t-1}^1, c_{t-1}^1))\\
# (h_t^2, c_t^2) &= \text{EncoderLSTM}^2(h_t^1, (h_{t-1}^2, c_{t-1}^2))
# \end{align*}$$
#
# Note how only our hidden state from the first layer is passed as input to the second layer, and not the cell state.
#
# So our encoder looks something like this:
#
# ![](assets/seq2seq2.png)
#
# We create this in code by making an `Encoder` module, which requires we inherit from `torch.nn.Module` and use the
# `super().__init__()` as some boilerplate code. The encoder takes the following arguments: - `input_dim` is the
# size/dimensionality of the one-hot vectors that will be input to the encoder. This is equal to the input (source)
# vocabulary size. - `emb_dim` is the dimensionality of the embedding layer. This layer converts the one-hot vectors
# into dense vectors with `emb_dim` dimensions. - `hid_dim` is the dimensionality of the hidden and cell states. -
# `n_layers` is the number of layers in the RNN. - `dropout` is the amount of dropout to use. This is a
# regularization parameter to prevent overfitting. Check out [this](
# https://www.coursera.org/lecture/deep-neural-network/understanding-dropout-YaGbR) for more details about dropout.
#
# We aren't going to discuss the embedding layer in detail during these tutorials. All we need to know is that there
# is a step before the words - technically, the indexes of the words - are passed into the RNN, where the words are
# transformed into vectors. To read more about word embeddings, check these articles: [1](
# https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/),
# [2](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html),
# [3](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/),
# [4](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/).
#
# The embedding layer is created using `nn.Embedding`, the LSTM with `nn.LSTM` and a dropout layer with `nn.Dropout`.
# Check the PyTorch [documentation](https://pytorch.org/docs/stable/nn.html) for more about these.
#
# One thing to note is that the `dropout` argument to the LSTM is how much dropout to apply between the layers of a
# multi-layer RNN, i.e. between the hidden states output from layer $l$ and those same hidden states being used for
# the input of layer $l+1$.
#
# In the `forward` method, we pass in the source sentence, $X$, which is converted into dense vectors using the
# `embedding` layer, and then dropout is applied. These embeddings are then passed into the RNN. As we pass a whole
# sequence to the RNN, it will automatically do the recurrent calculation of the hidden states over the whole
# sequence for us! Notice that we do not pass an initial hidden or cell state to the RNN. This is because,
# as noted in the [documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM), that if no hidden/cell
# state is passed to the RNN, it will automatically create an initial hidden/cell state as a tensor of all zeros.
#
# The RNN returns: `outputs` (the top-layer hidden state for each time-step), `hidden` (the final hidden state for
# each layer, $h_T$, stacked on top of each other) and `cell` (the final cell state for each layer, $c_T$,
# stacked on top of each other).
#
# As we only need the final hidden and cell states (to make our context vector), `forward` only returns `hidden` and
# `cell`.
#
# The sizes of each of the tensors is left as comments in the code. In this implementation `n_directions` will always
# be 1, however note that bidirectional RNNs (covered in tutorial 3) will have `n_directions` as 2.

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        # embedded = [src_len, batch size, emb dim]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


# ### Decoder
#
# Next, we'll build our decoder, which will also be a 2-layer (4 in the paper) LSTM.
#
# ![](assets/seq2seq3.png)
#
# The `Decoder` class does a single step of decoding, i.e. it ouputs single token per time-step. The first layer will
# receive a hidden and cell state from the previous time-step, $(s_{t-1}^1, c_{t-1}^1)$, and feeds it through the
# LSTM with the current embedded token, $y_t$, to produce a new hidden and cell state, $(s_t^1, c_t^1)$. The
# subsequent layers will use the hidden state from the layer below, $s_t^{l-1}$, and the previous hidden and cell
# states from their layer, $(s_{t-1}^l, c_{t-1}^l)$. This provides equations very similar to those in the encoder.
#
# $$\begin{align*}
# (s_t^1, c_t^1) = \text{DecoderLSTM}^1(d(y_t), (s_{t-1}^1, c_{t-1}^1))\\
# (s_t^2, c_t^2) = \text{DecoderLSTM}^2(s_t^1, (s_{t-1}^2, c_{t-1}^2))
# \end{align*}$$
#
# Remember that the initial hidden and cell states to our decoder are our context vectors, which are the final hidden
# and cell states of our encoder from the same layer, i.e. $(s_0^l,c_0^l)=z^l=(h_T^l,c_T^l)$.
#
# We then pass the hidden state from the top layer of the RNN, $s_t^L$, through a linear layer, $f$,
# to make a prediction of what the next token in the target (output) sequence should be, $\hat{y}_{t+1}$.
#
# $$\hat{y}_{t+1} = f(s_t^L)$$
#
# The arguments and initialization are similar to the `Encoder` class, except we now have an `output_dim` which is
# the size of the vocabulary for the output/target. There is also the addition of the `Linear` layer, used to make
# the predictions from the top layer hidden state.
#
# Within the `forward` method, we accept a batch of input tokens, previous hidden states and previous cell states. As
# we are only decoding one token at a time, the input tokens will always have a sequence length of 1. We `unsqueeze`
# the input tokens to add a sentence length dimension of 1. Then, similar to the encoder, we pass through an
# embedding layer and apply dropout. This batch of embedded tokens is then passed into the RNN with the previous
# hidden and cell states. This produces an `output` (hidden state from the top layer of the RNN), a new `hidden`
# state (one for each layer, stacked on top of each other) and a new `cell` state (also one per layer, stacked on top
# of each other). We then pass the `output` (after getting rid of the sentence length dimension) through the linear
# layer to receive our `prediction`. We then return the `prediction`, the new `hidden` state and the new `cell` state.
#
# **Note**: as we always have a sequence length of 1, we could use `nn.LSTMCell`, instead of `nn.LSTM`,
# as it is designed to handle a batch of inputs that aren't necessarily in a sequence. `nn.LSTMCell` is just a single
# cell and `nn.LSTM` is a wrapper around potentially multiple cells. Using the `nn.LSTMCell` in this case would mean
# we don't have to `unsqueeze` to add a fake sequence length dimension, but we would need one `nn.LSTMCell` per layer
# in the decoder and to ensure each `nn.LSTMCell` receives the correct initial hidden state from the encoder. All of
# this makes the code less concise - hence the decision to stick with the regular `nn.LSTM`.


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim  # 5893
        self.hid_dim = hid_dim  # 512
        self.n_layers = n_layers  # 2

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)  # 512, 5893

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]  [2, 64, 512]
        # cell = [n layers * n directions, batch size, hid dim]  [2, 64, 512]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # input = [1, batch size]
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)  # encoder作为context vector

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(
                input, hidden, cell)  # 把encoder的输出cell 和 hidden作为初始

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = len(SRC.vocab)  # 7854
OUTPUT_DIM = len(TRG.vocab)  # 5893
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS,
              ENC_DROPOUT)  # [7854, 256, 512, 2, 0.5]
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS,
              DEC_DROPOUT)  # [5893, 256, 512, 2, 0.5]

model = Seq2Seq(enc, dec, device).to(device)

model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 24
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(
        f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(
        f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
