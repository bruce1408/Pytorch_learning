import unicodedata
import re
import os
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# Configuring training
USE_CUDA = True
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
clip = 5.0
n_epochs = 50000
plot_every = 200
print_every = 1000
attn_model = 'concat'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
MAX_LENGTH = 10

good_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re "
)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./eng-fra/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')
    # Split every line into pairs and normalize

    pairs = [[normalize_string(s) for s in l.split('\t')[0:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p):
    """
    过滤函数,过滤掉每一个法英翻译对,最大长度不能大于10,且英文开头是带有 good_prefixes的data
    :param p:
    :return:
    """
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(good_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name, reverse=False):
    """
    数据准备阶段,
    :param lang1_name: 输入语种
    :param lang2_name: 输出语种
    :param reverse: 翻译翻转
    :return: 法英句子pairs
    """
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

# Print an example pair
print(random.choice(pairs))


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    """
    把句子变成index的list,然后再转化为tensor,最后加上一个1表示截止,shape是只有1列
    :param lang:
    :param sentence:
    :return:
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = torch.LongTensor(indexes).view(-1, 1)
    #     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var


def variables_from_pair(pair):
    """
    把输入和输出的句子对,进行index化,然后输出,得到的是一个str2index的tensor向量
    :param pair:
    :return:
    """
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return input_variable, target_variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)  # seq_len, batch, hidden_size
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


# class BahdanauAttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
#         super(AttnDecoderRNN, self).__init__()
#
#         # Define parameters
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p
#         self.max_length = MAX_LENGTH
#
#         # Define layers
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.dropout = nn.Dropout(dropout_p)
#         self.attn = Attn("concat", hidden_size)
#         self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
#         self.out = nn.Linear(hidden_size, output_size)
#
#     def forward(self, word_input, last_hidden, encoder_outputs):
#         # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
#
#         # Get the embedding of the current input word (last output word)
#         word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N
#         word_embedded = self.dropout(word_embedded)
#
#         # Quantity attention weights and apply to encoder outputs
#         attn_weights = self.attn(last_hidden[-1], encoder_outputs)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
#
#         # Combine embedded input word and attended context, run through RNN
#         rnn_input = torch.cat((word_embedded, context), 2)
#         output, hidden = self.gru(rnn_input, last_hidden)
#
#         # Final output layer
#         output = output.squeeze(0)  # B x N
#         output = F.log_softmax(self.out(torch.cat((output, context), 1)))
#
#         # Return final output, hidden state, and attention weights (for visualization)
#         return output, hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = torch.zeros(seq_len)  # B x 1 x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Quantity energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':  # ht * hs
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':  # ht * wa * hs
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
        elif self.method == 'concat':  # wa * [ht, hs] 拼接结果
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy


class DecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attention(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        """
        :param word_input: decoder_input
        :param last_context: decoder_context
        :param last_hidden: decoder_hidden
        :param encoder_outputs: decoder_outputs
        :return:
        """
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)

        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Quantity attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N

        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


# def testModel():
#     encoder_test = EncoderRNN(10, 10, 2)
#     decoder_test = AttnDecoderRNN('general', 10, 10, 2)
#     print(encoder_test)
#     print(decoder_test)
#
#     encoder_hidden = encoder_test.init_hidden()
#     word_input = torch.LongTensor([1, 2, 3])
#     if USE_CUDA:
#         encoder_test.cuda()
#         word_input = word_input.cuda()
#     encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)
#
#     word_inputs = Variable(torch.LongTensor([1, 2, 3]))
#     decoder_attns = torch.zeros(1, 3, 3)
#     decoder_hidden = encoder_hidden
#     decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))
#
#     if USE_CUDA:
#         decoder_test.cuda()
#         word_inputs = word_inputs.cuda()
#         decoder_context = decoder_context.cuda()
#
#     for i in range(3):
#         decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(
#             word_inputs[i],
#             decoder_context,
#             decoder_hidden,
#             encoder_outputs)
#         print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
#         decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().Dataset


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_context = torch.zeros(1, decoder.hidden_size)
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = DecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

# Begin to train!
for epoch in range(1, n_epochs + 1):

    # Get training Dataset for this cycle
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                 MAX_LENGTH)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0:
        continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
