# Deep Learning Notes for Natural Language Processing
## Steps to build a neural notework through Pytorch
### Preprocess the data
#### Build a counter
```python
import collections

# words=["hi", "hello", "fine", "hi"]
counter = collections.Counter(words)
# print(counter)
# Counter({'hi': 2, 'hello': 1, 'fine': 1})
```
#### Build a vocabulary
```python
import torchtext.vocab as Vocab

# Filter the word appearing less than 5 times
# Vocab is like a dictionary
vocab=Vocab.Vocab(counter, min_freq = 5)
```
#### Build tensors
```python
# We can use the word index to build tensors

# Pad the sentence to a fixed length
# Say maxLen = 500
def pad(x):
        return x[:maxLen] if len(x) > maxLen else x + [0] * (maxLen - len(x))

tensors = torch.tensor([pad([vocab.stoi[word] for word in sentence]) for sentence in paragraphs])
```
#### Initialize tensor dataset and data iterator
```python
import torch.utils.data as Data

tensorDataset = Data.TensorDataset(dataTensor, targetTensor)
```
Data Iterator is initialized in this way.
```python
# shuffle means manage the position of the given data in random
dataIter = Data.DataLoader(tensorDataset, batchSize, shuffle = True)
```
### Using nn.Module
* Define a nn.Mudule
```python
import torch # The torch version here is 1.6.0
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
    def forword(self, input):
        return output
```
Let's take Recurrent Neural Network as an example.
```python
class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, vocabulary, embeddingSize, numHiddens, numLayers):
        super(RecurrentNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(len(vocabulary), embeddingSize)
        self.encoder = nn.LSTM(input_size = embeddingSize,
                                hidden_size = numHiddens,
                                num_layers = numLayers,
                                bidirectional = True)
        self.decoder = nn.Linear(4 * numHiddens, 2)
    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        output = self.decoder(encoding)
        return output

embeddingSize, numHiddens, numLayers = 100, 100, 2
net = RecurrentNeuralNetwork(vocabulary, embeddingSize, numHiddens, numLayers)
```
#### Use pretrained GloVe (Recommended)
```python
gloveVocab = Vocab.GloVe(name = '6B', dim = 100, cache = os.path.join(ROOT, "Glove"))

# Filter out the words not in our dataset
def loadPretrainedEmbedding(words, pretrainedVocab):
    embed = torch.zeros(len(words), pretrainedVocab.vectors[0].shape[0])
    oovCnt = 0
    for i, word in enumerate(words):
        try:
            idx = pretrainedVocab.stoi[word]
            embed [i, :] = pretrainedVocab.vectors[idx]
        except KeyError:
            oovCnt += 1
    if oovCnt > 0:
        print('There are %d out of vocabulary words.' % oovCnt)
    return embed

net.embedding.weight.data.copy_(loadPretrainedEmbedding(vocab.itos, gloveVocab))
net.embedding.weight.requires_grad = False
```
#### Let's try to initialize this network on GPU
```python
print(torch.cuda.is_available()) # True
# x = torch.tensor([1,2,3]) # For now x is stored in memory.
# x.cuda() # x is moved to GPU memory now.
# I would recommend initializing the tensor in the following way.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device = device)
```
### Define Loss Function
```python
# Use Cross entropy loss function (Recommended)
lossFunction = nn.CrossEntropyLoss()
```
### Define an Optimizer
```python
# Use Adam optimizer (Recommeded)
optimizer = torch.optim.Adam(_, lr = learningRate)
```
### Training the Network
```python
import time
learningRate, numEpochs = 0.01, 5
# When we use this training function, let's assume everything is not moved to GPU yet.
def evaluateAccuracy(testIter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    accSum, n = 0, 0
    with torch.no_grad():
        for x, y in testIter:
            net.eval()
            accSum += (net(x.to(device)).argmax(dim = 1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return accSum / n
```
```python
def train(trainIter, testIter, net, loss, optimizer, device, numEpochs):
    net = net.to(device)
    print("We are training the data on ", device)
    batchCnt = 0
    for epoch in range(numEpochs):
        trainLossSum, trainAccSum, n, startTime = 0.0, 0.0, 0, time.time()
        for x, y in trainIter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            trainLossSum += l.cpu().item()
            trainAccSum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batchCnt += 1
        testAcc=evaluateAccuracy(testIter, net)
        print('epoch %d, loss %.3f, trainning accuracy %.3f, test accuracy %.3f, time %.1f sec' % (epoch + 1, trainLossSum / batchCnt, trainAccSum / n, testAcc, time.time() - startTime))
```
### Save the trained network on disk
```python
PATH = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/CNNIMDB.pt"
torch.save(net.state_dict(), PATH)
```
### Load the trained network on disk
```python
model = torch.load(PATH)
```
## A Simple Convolution Neural Network from Pytorch Tutorials
### A typical training procedure for a neural network is as follows:
* Define the neural network that has some learnable parameters (or weights)
* Iterate over a dataset of inputs
* Process input through the network
* Compute the loss (how far is the output from being correct)
* Propagate gradients back into the network’s parameters
* Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
#   (fc1): Linear(in_features=576, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
# 10
# torch.Size([6, 1, 3, 3])

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# tensor([[-0.0809, -0.0171,  0.0742,  0.0556,  0.1385, -0.0673,  0.0989, -0.0438, 0.1052,  0.1685]], grad_fn=<AddmmBackward>)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
# tensor(0.6939, grad_fn=<MseLossBackward>)
# The network works like this.
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# <MseLossBackward object at 0x7f3494a4fbe0>
# <AddmmBackward object at 0x7f3494a4fe48>
# <AccumulateGrad object at 0x7f3494a4fe48>

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# conv1.bias.grad before backward
# tensor([0., 0., 0., 0., 0., 0.])
# conv1.bias.grad after backward
# tensor([ 0.0206, -0.0234, -0.0080,  0.0055,  0.0006, -0.0060])

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```
This is one loop of a convolution neural network.
## Word2Vec Implement
```python
import torch
import numpy as np
import torch.nn as nn
import math
import collections
import random
import torch.utils.data as Data

# Preprocess the dataset
with open('C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    rawDataset=[line.split() for line in lines]

# See what is inside the given dataset
len(rawDataset)
for line in rawDataset[:3]:
    print(line)

# We only keep the word appearing for more than 5 times in our new dictionary
counter = collections.Counter([word for line in rawDataset for word in line])
counter = dict(filter(lambda x:x[1] >= 5, counter.items()))

# Build index and dataset
idx2word = [word for word, _ in counter.items()]
word2idx = {word: idx for idx, word in enumerate(idx2word)}
dataset = [[word2idx[word] for word in line if word in word2idx] for line in rawDataset]

# The sum of words in the dataset
wordSum = sum([len(line) for line in dataset])
print(wordSum)

# Discard high-frequency words for better performance
def discard(idx):
    return random.uniform(0,1) < 1-math.sqrt(1e-4 / counter[idx2word[idx]] * wordSum)
subDataset=[[word for word in line if not discard(word)] for line in dataset]

# The sum of words after discard
print(sum([len(line) for line in subDataset]))

def computeCenterNContext(dataset, maxWinSize):
    centers, contexts = [], []
    for line in dataset:
        if len(line) < 2:
            continue
        centers += line
        for center in range(len(line)):
            winSize=random.randint(1, maxWinSize)
            indices=list(range(max(0, center - winSize),min(len(line),center + 1 + winSize)))
            indices.remove(center)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

centers, contexts = computeCenterNContext(subDataset, 5)

# Negative sampling
def computeNegs(contexts, samplingWeights, K):
    allNegs, negCandidates, i = [], [], 0
    population = list(range(len(samplingWeights)))
    for cons in contexts:
        negs = []
        while len(negs) < len(cons)*K:
            if i ==l en(negCandidates):
                i, negCandidates = 0, random.choices(population, samplingWeights, k=int(1e5))
            neg, i = negCandidates[i], i+1
            if neg not in set(cons):
                negs.append(neg)
        allNegs.append(negs)
    return allNegs

samplingWeights = [counter[w] ** 0.75 for w in idx2word]
allNegs=computeNegs(contexts, samplingWeights, 5)

# Build the torch dataset
class DataSet1(Data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers)==len(contexts)==len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
        
    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])
    
    def __len__(self):
        return len(self.centers)

def batchify(data):
    maxLen = max(len(c)+len(n) for _, c, n in data)
    centers, contextsNegatives, masks, labels = [],[],[],[]
    for center, context, negative in data:
        curLen=len(context)+len(negative)
        centers += [center]
        contextsNegatives += [context+negative+[0]*(maxLen-curLen)]
        masks += [[1] * curLen+[0] * (maxLen - curLen)]
        labels +=  [[1] * len(context)+[0] * (maxLen - len(context))]
    return (torch.tensor(centers).view(-1,1), torch.tensor(contextsNegatives), torch.tensor(masks),torch.tensor(labels))

batchSize = 512 
finalDataset=DataSet1(centers, contexts, allNegs)

dataIter=Data.DataLoader(finalDataset, batchSize, shuffle = True, collate_fn = batchify, num_workers = 0)

for batch in dataIter:
    for name, data in zip(['centers', 'contextNegatives', 'masks', 'labels'], batch):
        print(name, 'shape:', data.shape)
    break

# The skip gram model
def skipGram(center, contextsNNegatives, embedV, embedU):
    v = embedV(center)
    u = embedU(contextsNNegatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))# bmm is something like multiplying matrixes
    return pred

class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res=nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction = "none", weight = mask)
        return res.mean(dim = 1)
    
loss=SigmoidBinaryCrossEntropyLoss()

embedSize = 100

# The net is defined here
net=nn.Sequential(nn.Embedding(num_embeddings = len(idx2word), embedding_dim = embedSize), nn.Embedding(num_embeddings = len(idx2word), embedding_dim = embedSize))

def train(net, lr, numEpochs):
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(numEpochs):
        start, lSum, n = time.time(), 0.0, 0
        for batch in dataIter:
            center, contextNegative, mask, label = [d.cuda() for d in batch]
            pred=skipGram(center, contextNegative, net[0], net[1])
            l=(loss(pred.view(label.shape), label, mask) * mask.shape[1] / mask.float().sum(dim = 1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            lSum += l.cpu().item()
            n += 1
        print("epoch %d, loss %.3f, time %.3fs" % (epoch + 1, lSum / n, time.time() - start))

# Application
def getSimilarTokens(queryToken, k, embed):
    W = embed.weight.data
    x = W[word2idx[queryToken]]
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim = 1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topK = torch.topk(cos, k=k+1)
    topK=topK.cpu().numpy()
    for i in topK[1:]:
        print("cosine sim = %.3f: %s"%(cos[i], idx2word[i]))

getSimilarTokens('chip', 3, net[0])
```
## Recurrent Neural Network
```python
import collections
import os
import random
import tarfile
import torch
import torch.nn as nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm
import time

device = torch.device('cuda')

# Preprocess the data
ROOT = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data"
fileName = os.path.join(ROOT, "aclImdb_v1.tar.gz")
if not os.path.exists(os.path.join(ROOT, "aclImdb")):
    print("Unzipping...")
    with tarfile.open(fileName, 'r') as f:
        f.extractall(ROOT)
print("Unzipping is done")

# 1 for positve and 0 for negative
def readImdb(folder = 'train', dataRoot = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/aclImdb"):
    data = []
    for label in ['pos', 'neg']:
        folderName = os.path.join(dataRoot, folder, label)
        for file in tqdm(os.listdir(folderName)):
            with open(os.path.join(folderName, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

trainData, testData = readImdb('train'), readImdb('test')

# Split the words by Blankspace
def getTokenizedImdb(data):
    def tokenizer(text):
        return [token.lower() for token in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

# Filter the words appearing less than 5 times
def getVocabImdb(data):
    tokenizedData = getTokenizedImdb(data)
    counter = collections.Counter([token for sentense in tokenizedData for token in sentense])
    return Vocab.Vocab(counter, min_freq = 5)

vocab = getVocabImdb(trainData)
len(vocab)

# Set the length of each comment to 500
# Padding is used here
def preprocessImdb(data, vocab):
    maxLen = 500
    def pad(x):
        return x[:maxLen] if len(x) > maxLen else x + [0] * (maxLen - len(x))
    tokenizedData = getTokenizedImdb(data)
    features=torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenizedData])
    labels = torch.tensor([score for _, score in data])
    return features, labels

batchSize = 64
trainSet = Data.TensorDataset(*preprocessImdb(trainData, vocab))
testSet = Data.TensorDataset(*preprocessImdb(testData, vocab))
trainIter = Data.DataLoader(trainSet, batchSize, shuffle = True)
testIter = Data.DataLoader(testSet, batchSize)

for x, y in trainIter:
    print('x', x.shape, 'y', y.shape)
    break
print(len(trainIter))

# The Bidirectional RNN model is defined here
class RNN(nn.Module):
    def __init__(self, vocab, embedSize, numHiddens, numLayers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedSize)
        self.encoder = nn.LSTM(input_size = embedSize,
                              hidden_size = numHiddens,
                              num_layers = numLayers,
                              bidirectional = True)
        self.decoder = nn.Linear(4 * numHiddens, 2)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

embedSize, numHiddens, numLayers = 100, 100, 2
net = RNN(vocab, embedSize, numHiddens, numLayers)
print(net)

# Donwnload the pretrained vocabulary from the website
gloveVocab = Vocab.GloVe(name = '6B', dim = 100, cache = os.path.join(ROOT, "Glove"))

# Filter the words not appeared in the comments
def loadPretrainedEmbedding(words, pretrainedVocab):
    embed = torch.zeros(len(words), pretrainedVocab.vectors[0].shape[0])
    oovCnt = 0
    for i, word in enumerate(words):
        try:
            idx = pretrainedVocab.stoi[word]
            embed [i, :] = pretrainedVocab.vectors[idx]
        except KeyError:
            oovCnt += 1
    if oovCnt > 0:
        print('There are %d out of vocabulary words.' % oovCnt)
    return embed

# Because the vocubulary data is pretrained, we do not need to update it
net.embedding.weight.data.copy_(loadPretrainedEmbedding(vocab.itos, gloveVocab))
net.embedding.weight.requires_grad = False

# Filter the embedding parameters which do not compute gradients
lr, numEpoch = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

def evaluateAccuracy(testIter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    accSum, n = 0, 0
    with torch.no_grad():
        for x, y in testIter:
            net.eval()
            accSum += (net(x.to(device)).argmax(dim = 1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return accSum / n

def train(trainIter, testIter, net, loss, optimizer, device, numEpochs):
    net = net.to(device)
    print("We are training the data on", device)
    batchCnt = 0
    for epoch in range(numEpochs):
        trainLossSum, trainAccSum, n, startTime = 0.0, 0.0, 0, time.time()
        for x, y in trainIter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            trainLossSum += l.cpu().item()
            trainAccSum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batchCnt += 1
        testAcc=evaluateAccuracy(testIter, net)
        print('epoch %d, loss %.3f, trainning accuracy %.3f, test accuracy %.3f, time %.1f sec' % (epoch + 1, trainLossSum / batchCnt, trainAccSum / n, testAcc, time.time() - startTime))

train(trainIter, testIter, net, loss, optimizer, device, numEpoch)

# Application: predict the emotion of the comment
def predictSentiment(net, vocab, sentence):
    device = torch.device('cuda')
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device = device)
    label = torch.argmax(net(sentence.view(1, -1)), dim = 1)
    return 'positive' if label.item() == 1 else 'negative'

# sentiment = ['i', 'like', 'the', 'story']
# sentiment = ['i', 'hate', 'the', 'story']
# sentiment = ['hate']
# sentiment = ['the', 'story', 'is', 'complex', 'but', 'i', 'love', 'it']
# sentiment = ['the', 'story', 'is', 'complex']
# sentiment = ['i', 'like', 'the', 'story', 'but', 'i', 'hate', 'the', 'character']
predictSentiment(net, vocab, sentiment)

# Save the trained network on disk
PATH = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/RNNIMDB.pt"
torch.save(net.state_dict(), PATH)
```
## Convolutional Nerual Network
```python
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import random
import collections
import torchtext.vocab as Vocab
from tqdm import tqdm
import time

device = torch.device('cuda')
ROOT = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data"

# Preprocess the data
def readImdb(folder = 'train', dataRoot = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/aclImdb"):
    data = []
    for label in ['pos', 'neg']:
        folderName = os.path.join(dataRoot, folder, label)
        for file in tqdm(os.listdir(folderName)):
            with open(os.path.join(folderName, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

def getTokenizedImdb(data):
    def tokenizer(text):
        return [token.lower() for token in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def getVocabImdb(data):
    tokenizedData = getTokenizedImdb(data)
    counter = collections.Counter([token for sentense in tokenizedData for token in sentense])
    return Vocab.Vocab(counter, min_freq = 5)

def preprocessImdb(data, vocab):
    maxLen = 500
    def pad(x):
        return x[:maxLen] if len(x) > maxLen else x + [0] * (maxLen - len(x))
    tokenizedData = getTokenizedImdb(data)
    features=torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenizedData])
    labels = torch.tensor([score for _, score in data])
    return features, labels

def loadPretrainedEmbedding(words, pretrainedVocab):
    embed = torch.zeros(len(words), pretrainedVocab.vectors[0].shape[0])
    oovCnt = 0
    for i, word in enumerate(words):
        try:
            idx = pretrainedVocab.stoi[word]
            embed [i, :] = pretrainedVocab.vectors[idx]
        except KeyError:
            oovCnt += 1
    if oovCnt > 0:
        print('There are %d out of vocabulary words.' % oovCnt)
    return embed

def evaluateAccuracy(testIter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    accSum, n = 0, 0
    with torch.no_grad():
        for x, y in testIter:
            net.eval()
            accSum += (net(x.to(device)).argmax(dim = 1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return accSum / n

def train(trainIter, testIter, net, loss, optimizer, device, numEpochs):
    net = net.to(device)
    print("We are training the data on", device)
    batchCnt = 0
    for epoch in range(numEpochs):
        trainLossSum, trainAccSum, n, startTime = 0.0, 0.0, 0, time.time()
        for x, y in trainIter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            trainLossSum += l.cpu().item()
            trainAccSum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batchCnt += 1
        testAcc=evaluateAccuracy(testIter, net)
        print('epoch %d, loss %.3f, trainning accuracy %.3f, test accuracy %.3f, time %.1f sec' % (epoch + 1, trainLossSum / batchCnt, trainAccSum / n, testAcc, time.time() - startTime))

# THe network is defined here
class GlobalMaxPool(nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size = x.shape[2])
    
class CNN(nn.Module):
    def __init__(self, vocab, embedSize, kernelSizes, numChannels):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedSize)
        self.constant_embedding = nn.Embedding(len(vocab), embedSize)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(numChannels), 2)
        self.pool = GlobalMaxPool()
        self.convs = nn.ModuleList()
        for c, k in zip(numChannels, kernelSizes):
            self.convs.append(nn.Conv1d(in_channels = 2 * embedSize,
                                       out_channels = c,
                                       kernel_size = k))
    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs), 
                               self.constant_embedding(inputs)), dim = 2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim = 1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

trainData, testData = readImdb('train'), readImdb('test')

batchSize = 64

vocab = getVocabImdb(trainData)
len(vocab)

embedSize, kernelSizes, numChannels = 100, [3, 4, 5], [100, 100, 100]
net = CNN(vocab, embedSize, kernelSizes, numChannels)

trainSet = Data.TensorDataset(*preprocessImdb(trainData, vocab))
testSet = Data.TensorDataset(*preprocessImdb(testData, vocab))
trainIter = Data.DataLoader(trainSet, batchSize, shuffle = True)
testIter = Data.DataLoader(testSet, batchSize)

gloveVocab = Vocab.GloVe(name = '6B', dim = 100, cache = os.path.join(ROOT, "Glove"))

lr, numEpoch = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

net.embedding.weight.data.copy_(loadPretrainedEmbedding(vocab.itos, gloveVocab))
net.constant_embedding.weight.requires_grad = False

train(trainIter, testIter, net, loss, optimizer, device, numEpoch)

def predictSentiment(net, vocab, sentence):
    device = torch.device('cuda')
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device = device)
    label = torch.argmax(net(sentence.view(1, -1)), dim = 1)
    return 'positive' if label.item() == 1 else 'negative'

# inputs.size must be longer than 5 here. (greater than kernel size)
# sentiment = ['the', 'story', 'is', 'complex', 'but', 'i', 'love', 'it']
# sentiment = ['the', 'story', 'is', 'complex']
# sentiment = ['i', 'like', 'the', 'story', 'but', 'i', 'hate', 'the', 'character']
predictSentiment(net, vocab, sentiment)

PATH = "C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/CNNIMDB.pt"
torch.save(net.state_dict(), PATH)
```
## Seq2Seq Translation Model
```python
import collections
import os
import io
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
import time

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
device = torch.device('cuda')

# Preprocess the data
def processOneSeq(seqTokens, allTokens, allSeqs, maxSeqLen):
    allTokens.extend(seqTokens)
    seqTokens += [EOS] + [PAD] * (maxSeqLen - len(seqTokens) - 1)
    allSeqs.append(seqTokens)
    
def buildData(allTokens, allSeqs):
    vocab = Vocab.Vocab(collections.Counter(allTokens), specials = [PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in allSeqs]
    return vocab, torch.tensor(indices)

def readData(maxSeqLen):
    inTokens, outTokens, inSeqs, outSeqs = [], [], [], []
    with io.open('C:/Users/gazla/Documents/STUDY/Research/d2l-zh/data/fr-en-small.txt') as f:
        lines = f.readlines()
        for line in lines:
            # Seperate the French and English
            inSeq, outSeq = line.rstrip().split('\t')
            inSeqTokens, outSeqTokens = inSeq.split(' '), outSeq.split(' ')
            if max(len(inSeqTokens), len(outSeqTokens)) > maxSeqLen - 1:
                continue
            processOneSeq(inSeqTokens, inTokens, inSeqs, maxSeqLen)
            processOneSeq(outSeqTokens, outTokens, outSeqs, maxSeqLen)
        inVocab, inData = buildData(inTokens, inSeqs)
        outVocab, outData = buildData(outTokens, outSeqs)
        return inVocab, outVocab, Data.TensorDataset(inData, outData)

maxSeqLen = 7
inVocab, outVocab, dataset = readData(maxSeqLen)
# The index of the words
dataset[0]

# The encoder is defined here
class Encoder(nn.Module):
    def __init__(self, vocabSize, embedSize, numHiddens, numLayers, dropRate = 0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocabSize, embedSize)
        self.rnn = nn.GRU(embedSize, numHiddens, numLayers, dropout = dropRate)
    
    def forward(self, inputs, state):
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        return self.rnn(embedding, state)
    
    def begin_state(self):
        return None

encoder = Encoder(vocabSize = 10, embedSize = 8, numHiddens = 16, numLayers = 2)
output, state = encoder(torch.zeros((4, 7)), encoder.begin_state())
output.shape, state.shape

# Focus on different parts on different states
def attentionModel(inputSize, attentionSize):
    model = nn.Sequential(nn.Linear(inputSize, attentionSize, bias = False),
                         nn.Tanh(), 
                         nn.Linear(attentionSize, 1, bias = False))
    return model

def attentionForward(model, encStates, decState):
    # Set to the same size before connecting them
    decStates = decState.unsqueeze(dim = 0).expand_as(encStates)
    encDecStates = torch.cat((encStates, decStates), dim = 2)
    e = model(encDecStates)
    alpha = F.softmax(e, dim = 0)
    return (alpha * encStates).sum(dim = 0)

seqLen, batchSize, numHiddens = 10, 4, 8
model = attentionModel(2 * numHiddens, 10)
encStates = torch.zeros((seqLen, batchSize, numHiddens))
decState = torch.zeros((batchSize, numHiddens))
attentionForward(model, encStates, decState).shape

# The decoder is defined here
class Decoder(nn.Module):
    def __init__(self, vocabSize, embedSize, numHiddens, numLayers, attentionSize, dropRate = 0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embedSize)
        self.attention = attentionModel(2 * numHiddens, attentionSize)
        self.rnn = nn.GRU(2 * embedSize, numHiddens, numLayers, dropout = dropRate)
        self.out = nn.Linear(numHiddens, vocabSize)
        
    def forward(self, curInput, state, encStates):
        c = attentionForward(self.attention, encStates, state[-1])
        inputC = torch.cat((self.embedding(curInput), c), dim = 1)
        output, state = self.rnn(inputC.unsqueeze(0), state)
        output = self.out(output).squeeze(dim = 0)
        return output, state
    
    def begin_state(self, encState):
        # The initial state of the decoder is the final state of the encoder
        return encState

def batchLoss(encoder, decoder, X, Y, loss):
    batchSize = X.shape[0]
    encState = encoder.begin_state()
    encOutputs, encState = encoder(X, encState)
    decState = decoder.begin_state(encState)
    decInput = torch.tensor([outVocab.stoi[BOS]] * batchSize)
    mask, numNotPadTokens = torch.ones(batchSize, ), 0
    l = torch.tensor([0.0])
    for y in Y.permute(1, 0):
        decOutput, decState = decoder(decInput, decState, encOutputs)
        l = l + (mask *  loss(decOutput, y)).sum()
        decInput = y # Force teaching
        numNotPadTokens += mask.sum().item()
        mask = mask * (y != outVocab.stoi[PAD]).float()
    return l / numNotPadTokens

def train(encoder, decoder, dataset, lr, batchSize, numEpochs):
    encOptimizer = torch.optim.Adam(encoder.parameters(), lr = lr)
    decOptimizer = torch.optim.Adam(decoder.parameters(), lr = lr)
    startTime = time.time()
    loss = nn.CrossEntropyLoss(reduction = 'none')
    dataIter = Data.DataLoader(dataset, batchSize, shuffle = True)
    for epoch in range(numEpochs):
        lSum = 0.0
        for X, Y in dataIter:
            encOptimizer.zero_grad()
            decOptimizer.zero_grad()
            l = batchLoss(encoder, decoder, X, Y, loss)
            l.backward()
            encOptimizer.step()
            decOptimizer.step()
            lSum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f, time %.3f sec" % (epoch + 1, lSum / len(dataIter), time.time() - startTime))

embedSize, numHiddens, numLayers = 64, 64, 2
attentionSize, dropRate, lr, batchSize, numEpochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(inVocab), embedSize, numHiddens, numLayers, dropRate)
decoder = Decoder(len(outVocab), embedSize, numHiddens, numLayers, attentionSize, dropRate)

train(encoder, decoder, dataset, lr, batchSize, numEpochs)

# Application: translation
# Greedy Searching is used here
def translate(encoder, decoder, inputSeq, maxSeqLen):
    inTokens = inputSeq.split(' ')
    inTokens += [EOS] + [PAD] * (maxSeqLen - len(inTokens) - 1)
    encInput = torch.tensor([[inVocab.stoi[tk] for tk in inTokens]])
    encState = encoder.begin_state()
    encOutput, encState = encoder(encInput, encState)
    decInput = torch.tensor([outVocab.stoi[BOS]])
    decState = decoder.begin_state(encState)
    outputTokens = []
    for _ in range(maxSeqLen):
        decOutput, decState = decoder(decInput, decState, encOutput)
        pred = decOutput.argmax(dim = 1)
        predToken = outVocab.itos[int(pred.item())]
        if predToken == EOS:
            break
        else: 
            outputTokens.append(predToken)
            decInput= pred
    return outputTokens

# inputSeq = 'ils regardent'
# inputSeq = 'bonjour' INVALID_INPUT
# inputSeq = 'cela me dit rien'
# This is a very small dataset, and it stores keywords in a dictionary.
# For the words which cannot be found in the dictionary, the model cannot translate them.
# Mind the dataset is quite small
translate(encoder, decoder, inputSeq, maxSeqLen)

# Bleu Estimation
def bleu(predTokens, labelTokens, k):
    lenPred, lenLabel = len(predTokens), len(labelTokens)
    score = math.exp(min(0, 1 - lenLabel / lenPred))
    for n in range(1, k + 1):
        numMatches, labelSubs = 0, collections.defaultdict(int)
        for i in range(lenLabel - n + 1):
            labelSubs[''.join(labelTokens[i: i + n])] += 1
        for i in range(lenPred - n + 1):
            if labelSubs[''.join(predTokens[i: i + n])] > 0:
                numMatches += 1
                labelSubs[''.join(predTokens[i: i + n])] -= 1
        score *= math.pow(numMatches / (lenPred - n + 1), math.pow(0.5, n))
    return score

def score(inqutSeq, labelSeq, k):
    predTokens = translate(encoder, decoder, inputSeq, maxSeqLen)
    labelTokens = labelSeq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(predTokens, labelTokens, k), ' '.join(predTokens)))

score('ils regardent .', 'they are watching .', k = 2)
```
# End
Special Thanks to Dive into Deep Learning and its Pytorch Version