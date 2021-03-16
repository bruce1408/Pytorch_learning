import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# define the simple model structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# define random Dataset
random_input = torch.randn(10,)
random_target = torch.randn(1,)

# ===== before train the model, the fc2 para is: =======
net = Net()
print(30*"=" + 'fc2 weight before train:' + 30 * "=")
print(net.fc2.weight)

# =========== begin to train the model ==================
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)
for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()

# ===== after train the model, fc2 has been changed =====
print(30*"="+'fc2 weight after train:' + "="*30)
print(net.fc2.weight)

# save the net
torch.save(net.state_dict(), 'model.pt')

# delete and redefine the net
del net
net = Net()

# load the weight
net.load_state_dict(torch.load('model.pt'))

# print the pre-trained fc2 weight
print(30*"=" + 'fc2 pretrained weight (same as the one above):'+20*"=")
print(net.fc2.weight)

# define new random Dataset
random_input = torch.randn(10,)
random_target = torch.randn(1,)

# we want to freeze the fc2 layer this time: only train fc1 and fc3
net.fc2.weight.requires_grad = False
net.fc2.bias.requires_grad = False

# train again
criterion = nn.MSELoss()

# NOTE: pytorch optimizer explicitly accepts parameter that requires grad
# see https://github.com/pytorch/pytorch/issues/679
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
# this raises ValueError: optimizing a parameter that doesn't require gradients

for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()

# print the retrained fc2 weight
# note that the weight is same as the one before retraining: only fc1 & fc3 changed
print(30*"="+'fc2 weight (frozen) after retrain:'+30*"=")
print(net.fc2.weight)

# let's unfreeze the fc2 layer this time for extra tuning
net.fc2.weight.requires_grad = True
net.fc2.bias.requires_grad = True

# add the unfrozen fc2 weight to the current optimizer
optimizer.add_param_group({'params': net.fc2.parameters()})

# re-retrain
for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()

# print the re-retrained fc2 weight
# note that this time the fc2 weight also changed
print(30*"="+'fc2 weight (unfrozen) after re-retrain:'+30*"=")
print(net.fc2.weight)