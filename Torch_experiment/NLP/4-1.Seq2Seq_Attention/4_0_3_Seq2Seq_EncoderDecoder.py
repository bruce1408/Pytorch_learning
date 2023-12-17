# coding: utf-8
# # 2 - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
# 
# In this second notebook on sequence-to-sequence models using PyTorch and TorchText, we'll be implementing the model
# from [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](
# https://arxiv.org/abs/1406.1078). This model will achieve improved test perplexity whilst only using a single layer
# RNN in both the encoder and the decoder.
# 
# ## Introduction
# 
# Let's remind ourselves of the general encoder-decoder model.
# 
# ![](assets/seq2seq1.png)
# 
# We use our encoder (green) over the embedded source sequence (yellow) to create a context vector (red). We then use
# that context vector with the decoder (blue) and a linear layer (purple) to generate the target sentence.
# 
# In the previous model, we used an multi-layered LSTM as the encoder and decoder.
# 
# ![](assets/seq2seq4.png)
# 
# One downside of the previous model is that the decoder is trying to cram lots of information into the hidden
# states. Whilst decoding, the hidden state will need to contain information about the whole of the source sequence,
# as well as all of the tokens have been decoded so far. By alleviating some of this information compression,
# we can create a better model!
# 
# We'll also be using a GRU (Gated Recurrent Unit) instead of an LSTM (Long Short-Term Memory). Why? Mainly because
# that's what they did in the paper (this paper also introduced GRUs) and also because we used LSTMs last time. To
# understand how GRUs (and LSTMs) differ from standard RNNS, check out [this](
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/) link. Is a GRU better than an LSTM? [Research](
# https://arxiv.org/abs/1412.3555) has shown they're pretty much the same, and both are better than standard RNNs.
# 
# ## Preparing Data
# 
# All of the Dataset preparation will be (almost) the same as last time, so we'll very briefly detail what each code
# block does. See the previous notebook for a recap.
# 
# We'll import PyTorch, TorchText, spaCy and a few standard modules.


import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
import math
import time

"""
paper: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
"""

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Instantiate our German and English spaCy models.


spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
# Previously we reversed the source (German) sentence, however in the paper we are implementing they don't do this,
# so neither will we.


# Previously we reversed the source (German) sentence, however in the paper we are implementing they don't do this,
# so neither will we.


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# Create our fields to process our Dataset. This will append the "start of sentence" and "end of sentence" tokens as
# well as converting all words to lowercase.


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

# Load our Dataset.


train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))

# We'll also print out an example just to double check they're not reversed.
print(vars(train_data.examples[0]))

# Then create our vocabulary, converting all tokens appearing less than twice into `<unk>` tokens.

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Finally, define the `device` and create our iterators.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)  # no cell state!

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


# ## Seq2Seq Model
# 
# Putting the encoder and decoder together, we get:
# 
# ![](assets/seq2seq7.png)
# 
# Again, in this implementation we need to ensure the hidden dimensions in both the encoder and the decoder are the
# same.
# 
# Briefly going over all of the steps: - the `outputs` tensor is created to hold all predictions, $\hat{Y}$ - the
# source sequence, $X$, is fed into the encoder to receive a `context` vector - the initial decoder hidden state is
# set to be the `context` vector, $s_0 = z = h_T$ - we use a batch of `<sos>` tokens as the first `input`,
# $y_1$ - we then decode within a loop: - inserting the input token $y_t$, previous hidden state, $s_{t-1}$,
# and the context vector, $z$, into the decoder - receiving a prediction, $\hat{y}_{t+1}$, and a new hidden state,
# $s_t$ - we then decide if we are going to teacher force or not, setting the next input as appropriate (either the
# ground truth next token in the target sequence or the highest predicted next token)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)  # context 这里一直不变

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


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device)


# Next, we initialize our parameters. The paper states the parameters are initialized from a normal distribution with
# a mean of 0 and a standard deviation of 0.01, i.e. $\mathcal{N}(0, 0.01)$.
# 
# It also states we should initialize the recurrent parameters to a special initialization, however to keep things
# simple we'll also initialize them to $\mathcal{N}(0, 0.01)$.


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


model.apply(init_weights)

# We print out the number of parameters.
# 
# Even though we only have a single layer RNN for our encoder and decoder we actually have **more** parameters  than
# the last model. This is due to the increased size of the inputs to the GRU and the linear layer. However,
# it is not a significant amount of parameters and causes a minimal amount of increase in training time (~3 seconds
# per epoch extra).


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# We initiaize our optimizer.


optimizer = optim.Adam(model.parameters())

# We also initialize the loss function, making sure to ignore the loss on `<pad>` tokens.


TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


# We then create the training loop...


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


# ...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing.


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


# We'll also define the function that calculates how long an epoch takes.


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Then, we train our model, saving the parameters that give us the best validation loss.


N_EPOCHS = 10
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
        torch.save(model.state_dict(), 'tut2-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Finally, we test the model on the test set using these "best" parameters.


model.load_state_dict(torch.load('tut2-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Just looking at the test loss, we get better performance. This is a pretty good sign that this model architecture
# is doing something right! Relieving the information compression seems like the way forard, and in the next tutorial
# we'll expand on this even further with *attention*.
