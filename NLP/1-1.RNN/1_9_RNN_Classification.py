# # Original code is from https://github.com/spro/practical-pytorch
# import os
# import time
# import math
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.utils.Dataset import DataLoader
#
# from utils.name_dataset import NameDataset
# from torch.nn.utils.rnn import pack_padded_sequence
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # Parameters and DataLoaders
# HIDDEN_SIZE = 100
# N_LAYERS = 2
# BATCH_SIZE = 256
# N_EPOCHS = 100
#
# test_dataset = NameDataset(is_train_set=False)
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=BATCH_SIZE, shuffle=True)
#
#
# train_dataset = NameDataset(is_train_set=True)
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=BATCH_SIZE, shuffle=True)
#
# N_COUNTRIES = len(train_dataset.get_countries())
# print(N_COUNTRIES, "countries")
# N_CHARS = 128  # ASCII
#
#
# # Some utility functions
# def time_since(since):
#     s = time.time() - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
#
#
# def create_variable(tensor):
#     # Do cuda() before wrapping with variable
#     if torch.cuda.is_available():
#         return Variable(tensor.cuda())
#     else:
#         return Variable(tensor)
#
#
# # pad sequences and sort the tensor
# def pad_sequences(vectorized_seqs, seq_lengths, countries):
#     seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
#     for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
#         seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
#
#     # Sort tensors by their length
#     seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#     seq_tensor = seq_tensor[perm_idx]
#
#     # Also sort the target (countries) in the same order
#     target = countries2tensor(countries)
#     if len(countries):
#         target = target[perm_idx]
#
#     # Return variables
#     # DataParallel requires everything to be a Variable
#     return create_variable(seq_tensor), create_variable(seq_lengths), create_variable(target)
#
#
# # Create necessary variables, lengths, and target
# def make_variables(names, countries):
#     sequence_and_length = [str2ascii_arr(name) for name in names]
#     vectorized_seqs = [sl[0] for sl in sequence_and_length]
#     seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
#     return pad_sequences(vectorized_seqs, seq_lengths, countries)
#
#
# def str2ascii_arr(msg):
#     arr = [ord(c) for c in msg]
#     return arr, len(arr)
#
#
# def countries2tensor(countries):
#     country_ids = [train_dataset.get_country_id(
#         country) for country in countries]
#     return torch.LongTensor(country_ids)
#
#
# class RNNClassifier(nn.Module):
#     # Our model
#
#     def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
#         super(RNNClassifier, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.n_directions = int(bidirectional) + 1
#
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
#                           bidirectional=bidirectional)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input, seq_lengths):
#         # Note: we run this all at once (over the whole input sequence)
#         # input shape: B x S (input size)
#         # transpose to make S(sequence) x B (batch)
#         input = input.t()
#         batch_size = input.size(1)
#
#         # Make a hidden
#         hidden = self._init_hidden(batch_size)
#
#         # Embedding S x B -> S x B x I (embedding size)
#         embedded = self.embedding(input)
#
#         # Pack them up nicely
#         gru_input = pack_padded_sequence(
#             embedded, seq_lengths.Dataset.cpu().numpy())
#
#         # To compact weights again call flatten_parameters().
#         self.gru.flatten_parameters()
#         output, hidden = self.gru(gru_input, hidden)
#
#         # Use the last layer output as FC's input
#         # No need to unpack, since we are going to use hidden
#         fc_output = self.fc(hidden[-1])
#         return fc_output
#
#     def _init_hidden(self, batch_size):
#         hidden = torch.zeros(self.n_layers * self.n_directions,
#                              batch_size, self.hidden_size)
#         return create_variable(hidden)
#
#
# # Train cycle
# def train():
#     total_loss = 0
#
#     for i, (names, countries) in enumerate(train_loader, 1):
#         input, seq_lengths, target = make_variables(names, countries)
#         output = classifier(input, seq_lengths)
#
#         loss = criterion(output, target)
#         total_loss += loss.Dataset
#
#         classifier.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if i % 10 == 0:
#             print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.2f}'.format(
#                 time_since(start), epoch,  i *
#                 len(names), len(train_loader.dataset),
#                 100. * i * len(names) / len(train_loader.dataset),
#                 total_loss / i * len(names)))
#
#     return total_loss
#
#
# # Testing cycle
# def test(name=None):
#     # Predict for a given name
#     if name:
#         input, seq_lengths, target = make_variables([name], [])
#         output = classifier(input, seq_lengths)
#         pred = output.Dataset.max(1, keepdim=True)[1]
#         country_id = pred.cpu().numpy()[0][0]
#         print(name, "is", train_dataset.get_country(country_id))
#         return
#
#     print("evaluating trained model ...")
#     correct = 0
#     train_data_size = len(test_loader.dataset)
#
#     for names, countries in test_loader:
#         input, seq_lengths, target = make_variables(names, countries)
#         output = classifier(input, seq_lengths)
#         pred = output.Dataset.max(1, keepdim=True)[1]
#         correct += pred.eq(target.Dataset.view_as(pred)).cpu().sum()
#
#     print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, train_data_size, 100. * correct / train_data_size))
#
#
# if __name__ == '__main__':
#
#     classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRIES, N_LAYERS)
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
#         classifier = nn.DataParallel(classifier)
#
#     if torch.cuda.is_available():
#         classifier.cuda()
#
#     optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#
#     start = time.time()
#     print("Training for %d epochs..." % N_EPOCHS)
#     for epoch in range(1, N_EPOCHS + 1):
#         # Train cycle
#         train()
#
#         # Testing
#         test()
#         #
#         # # Testing several samples
#         test("Sung")
#         test("Jungwoo")
#         test("Soojin")
#         test("Nako")


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/raid/bruce/tmp/pytorch_intro/Dataset/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/raid/bruce/tmp/pytorch_intro/Dataset/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)  # 双向LSTM
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  #

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step out是最后一层每一个时刻，把它变成最后一层的最后一个时刻的值。
        out = self.fc(out[:, -1, :])  # [100, 256]
        return self.softmax(out)


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
