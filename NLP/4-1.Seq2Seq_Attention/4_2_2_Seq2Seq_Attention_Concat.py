#!/usr/bin/env python
# coding: utf-8

# # 3 - Neural Machine Translation by Jointly Learning to Align and Translate
"""
In this third notebook on sequence-to-sequence models using PyTorch and TorchText,

we'll be implementing the model from

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

This model achives our best perplexity yet, ~27 compared to ~34 for the previous model.

## Introduction

As a reminder, here is the general encoder-decoder model:

In the previous model, our architecture was set-up in a way to reduce "information compression" by explicitly
passing the context vector, z, to the decoder at every time-step and by passing both the context vector and
embedded input word, d(y_t), along with the hidden state, s_t, to the linear layer, f, to make a prediction.


Even though we have reduced some of this compression, our context vector still needs to contain all of the
information about the source sentence. The model implemented in this notebook avoids this compression by allowing
the decoder to look at the entire source sentence (via its hidden states) at each decoding step! How does it do
this? It uses *attention*.

Attention works by first, calculating an attention vector, a, that is the length of the source sentence. The
attention vector has the property that each element is between 0 and 1, and the entire vector sums to 1. We then
calculate a weighted sum of our source sentence hidden states, $H$, to get a weighted source vector, w.

w = sum_{i} * a_ih_i

We calculate a new weighted source vector every time-step when decoding, using it as input to our decoder RNN as
well as the linear layer to make a prediction. We'll explain how to do all of this during the tutorial.

## Preparing Data

Again, the preparation is similar to last time.
"""

# First we import all the required modules.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

# Set the random seeds for reproducability.

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load the German and English spaCy models.
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


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


# The fields remain the same as before.

SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

# Load the Dataset.
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

# Build the vocabulary.
SRC.build_vocab(train_data, min_freq=2)

TRG.build_vocab(train_data, min_freq=2)

# Define the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the iterators.
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        """
        Encoder部分比较简单,就是使用的是一个双向GRU,hidden得到是 前向hidden1, 后向hidden2 拼接的一个多个层的hidden
        这里只选择最后一层的hidden,所以就是hidden[-1, :, :] 和 hidden[-2, :, :] 拼接之后再通过一个线性变换层得到的结果,这个结果再
        进行tanh函数来变换得到的就是最后需要喂给decoder的hidden,也是最为decoder部分初始的hidden.
        :param input_dim:
        :param emb_dim:
        :param enc_hid_dim:
        :param dec_hid_dim:
        :param dropout:
        """
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * num directions] = [23x128x1024]
        # hidden = [n layers * num directions, batch size, hid dim] = [2x128x512]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times -> [batch, src_len, dec_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


# ### Decoder
"""
Next up is the decoder. 

The decoder contains the attention layer, `attention`, which takes the previous hidden state, s_{t-1}, all of the 

encoder hidden states, $H$, and returns the attention vector, $a_t$.

We then use this attention vector to create a weighted source vector, $w_t$, denoted by `weighted`, 

which is a weighted sum of the encoder hidden states, $H$, using $a_t$ as the weights.

w_t = a_t * H

The embedded input word, d(y_t), the weighted source vector, w_t, and the previous decoder hidden state, 

s_{t-1}, are then all passed into the decoder RNN, with $d(y_t)$ and w_t being concatenated together.

s_t = text{DecoderGRU}(d(y_t), w_t, s_{t-1})

We then pass d(y_t), w_t and s_t through the linear layer, f, to make a prediction of the next word 

in the target sentence, hat{y}_{t+1}. This is done by concatenating them all together.

hat{y}_{t+1} = f(d(y_t), w_t, s_t), The image below shows decoding the first word in an example translation.


The green/teal blocks show the forward/backward encoder RNNs which output $H$, the red block shows the 
context vector, 

z = h_T = tanh(g(h^rightarrow_T, h^leftarrow_T)) = tanh(g(z^rightarrow, z^leftarrow)) = s_0, 

the blue block shows the decoder RNN which outputs s_t, the purple block shows the linear layer, f, 

which outputs hat{y}_{t+1} and the orange block shows the calculation of the weighted sum over H by a_t and 

outputs w_t. Not shown is the calculation of a_t.

"""


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        dot 计算的score是用当前的状态 hidden_t 计算当前的输出 yt, 而concat(Baha)的计算方式是使用上一个时刻的t来计算当前时刻的输出 yt
        input = [batch size]
        hidden = [batch size, dec hid dim] 第一个hidden 是 encoder 的 hidden
        encoder_outputs = [src len, batch size, enc hid dim * 2]
        :param input:
        :param hidden:
        :param encoder_outputs:
        :return:
        """

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)  # 计算attention部分

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)  # 计算context vector = ati * hi (hi 表示 encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # 添加一个全连接层去预测输出prediction

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


# Seq2Seq
"""
This is the first model where we don't have to have the encoder RNN and decoder RNN have the same hidden dimensions,

however the encoder has to be bidirectional. This requirement can be removed by changing all occurences

of `enc_dim * 2` to `enc_dim * 2 if encoder_is_bidirectional else enc_dim`.

This seq2seq encapsulator is similar to the last two. The only difference is that the `encoder` returns

both the final hidden state (which is the final hidden state from both the forward and backward encoder

RNNs passed through a linear layer) to be used as the initial hidden state for the decoder, as well as every

hidden state (which are the forward and backward hidden states stacked on top of each other). We also need to

ensure that `hidden` and `encoder_outputs` are passed to the decoder.

Briefly going over all of the steps:
- the `outputs` tensor is created to hold all predictions, hat{Y}
- the source sequence, $X$, is fed into the encoder to receive $z$ and $H$
- the initial decoder hidden state is set to be the `context` vector, $s_0 = z = h_T$
- we use a batch of `<sos>` tokens as the first `input`, $y_1$
- we then decode within a loop:
  - inserting the input token $y_t$, previous hidden state, $s_{t-1}$, and all encoder outputs, $H$, into the decoder
  - receiving a prediction, hat{y}_{t+1}, and a new hidden state, s_t
  - we then decide if we are going to teacher force or not, setting the next input as appropriate
"""


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        decoder_input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            decoder_input = trg[t] if teacher_force else top1

        return outputs


# ## Training the Seq2Seq Model
# The rest of this tutorial is very similar to the previous one.
# We initialise our parameters, encoder, decoder and seq2seq model (placing it on the GPU if we have one). 


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


# We use a simplified version of the weight initialization scheme used in the paper.
# Here, we will initialize all biases to zero and all weights from $\mathcal{N}(0, 0.01)$.
def init_weights(m):
    """
    初始化参数权重
    :param m:
    :return:
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)


# Quantity the number of parameters. We get an increase of almost 50% in the amount of parameters from the last model.


def count_parameters(model):
    """
    计算总共有多少参数需要计算
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# We create an optimizer.
optimizer = optim.Adam(model.parameters())

# We initialize the loss function.
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


# Finally, define a timing function.


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Then, we train our model, saving the parameters that give us the best validation loss.

N_EPOCHS = 5
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        print('valid loss is: ', valid_loss)
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}', end="")
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Finally, we test the model on the test set using these "best" parameters.

model.load_state_dict(torch.load('tut3-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# We've improved on the previous model, but this came at the cost of doubling the training time.
# 
# In the next notebook, we'll be using the same architecture but using a few tricks that are applicable
#
# to all RNN architectures - packed padded sequences and masking. We'll also implement code which will allow us to
#
# look at what words in the input the RNN is paying attention to when decoding the output.
