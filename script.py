import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import re
import random
import pickle
random.seed(1)
sst_home = 'data/trees'

# Let's do 2-way positive/negative classification instead of 5-way
easy_label_map = {0:0, 1:0, 2:None, 3:1, 4:1}
    # so labels of 0 and 1 in te 5-wayclassificaiton are 0 in the 2-way. 3 and 4 are 1, and 2 is none
    # because we don't have a neautral class. 

PADDING = "<PAD>"
UNKNOWN = "<UNK>"
max_seq_length = 20

def load_sst_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue
            
            # Strip out the parse information and the phrase labels---we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data
     
training_set = load_sst_data(sst_home + '/train.txt')
dev_set = load_sst_data(sst_home + '/dev.txt')
test_set = load_sst_data(sst_home + '/test.txt')

import collections
import numpy as np

def tokenize(string):
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
            example['label'] = torch.LongTensor([example['label']])


word_to_ix, vocab_size = build_dictionary([training_set])
sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set, test_set])


# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        batches.append(batch)
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels

def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = Variable(torch.stack(vectors).squeeze().cuda())
        labels = torch.stack(labels).squeeze().cuda()
        output = model(vectors)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / float(total)

# A Multi-Layer Perceptron (MLP)
class MLPClassifier(nn.Module): # inheriting from nn.Module!
    
    def __init__(self, input_size, embedding_dim, hidden_dim, num_labels, dropout_prob):
        super(MLPClassifier, self).__init__()
        
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout_prob)
            
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim) 
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, num_labels)
        self.init_weights()
        
    def forward(self, x):
        # Pass the input through your layers in order
        out = self.embed(x)
        out = self.dropout(out)
        out = torch.sum(out, dim=1)
        out = F.relu(self.linear_1(out))
        out = F.relu(self.linear_2(out))
        out = self.dropout(self.linear_3(out))
        return out

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_1, self.linear_2]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
def training_loop(model, loss, optimizer, training_iter, dev_iter, train_eval_iter):
    step = 0
    accs=[]
    for i in range(num_train_steps):
        model.train()
        vectors, labels = get_batch(next(training_iter))
        vectors = Variable(torch.stack(vectors).squeeze().cuda())
        labels = Variable(torch.stack(labels).squeeze().cuda())

        model.zero_grad()
        output = model(vectors)

        lossy = loss(output, labels)
        lossy.backward()
        optimizer.step()

        if step % 100 == 0:
            print( "Step %i; Loss %f; Train acc: %f; Dev acc %f" 
                %(step, lossy.data[0], evaluate(model, train_eval_iter), evaluate(model, dev_iter)))

        step += 1
        accs.append(evaluate(model, dev_iter))
    return accs
# Hyper Parameters 
input_size = vocab_size
num_labels = 2
batch_size = 32
num_train_steps = 1000

hidden_dim = 20
embedding_dim = 150
learning_rate = 0.001
dropout_prob = 0.3

def tuner(hidden_dim, embedding_dim, learning_rate, dropout_prob, varying_param, param_range, finalRes):
    def calc(hidden_dim, embedding_dim, learning_rate, dropout_prob, varying_param, param_value, finalRes):
        model = MLPClassifier(input_size, embedding_dim, hidden_dim, num_labels, dropout_prob)
        # Loss and Optimizer
        model.cuda()
        loss = nn.CrossEntropyLoss().cuda()  
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        training_iter = data_iter(training_set, batch_size)
        train_eval_iter = eval_iter(training_set[0:500], batch_size)
        dev_iter = eval_iter(dev_set[0:500], batch_size)
        finalRes[varying_param][param_value]=training_loop(model, loss, optimizer, training_iter, dev_iter, train_eval_iter)
    for r in param_range:
        if varying_param=="hidden_dim":
            calc(r, embedding_dim, learning_rate, dropout_prob, varying_param, r, finalRes)
        elif varying_param=="embedding_dim":
            calc(hidden_dim, r, learning_rate, dropout_prob, varying_param, r, finalRes)
        elif varying_param=="learning_rate":
            calc(hidden_dim, embedding_dim, r, dropout_prob, varying_param, r, finalRes)
        elif varying_param=="dropout_prob":
            calc(hidden_dim, embedding_dim, learning_rate, r, varying_param, r, finalRes)
# finalRes=collections.defaultdict(dict)
# print("Hidden Dim varying.")
# tuner(hidden_dim, embedding_dim, learning_rate, dropout_prob, 'hidden_dim', range(10,60, 5), finalRes)
# print("Embedding Dim varying.")
# tuner(hidden_dim, embedding_dim, learning_rate, dropout_prob, 'embedding_dim', range(50,400, 20), finalRes)
# print("Learning Rate varying.")
# tuner(hidden_dim, embedding_dim, learning_rate, dropout_prob, 'learning_rate',np.linspace(0.001, 0.1, num=10, endpoint=True)  , finalRes)
# print("Dropout Prob varying.")
# tuner(hidden_dim, embedding_dim, learning_rate, dropout_prob, 'dropout_prob', np.linspace(0.1, 0.9, num=9, endpoint=True) , finalRes)

# CBOW_file="am8676_cbow"
# fo=open(CBOW_file, 'wb')
# pickle.dump(finalRes, fo)
# fo.close()



class TextCNN(nn.Module):
    def __init__(self, input_size, embedding_dim, window_size, n_filters, num_labels, dropout_prob):
        super(TextCNN, self).__init__()
        
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p = dropout_prob)
        self.dropout2 = nn.Dropout(p = dropout_prob)
        self.conv1 = nn.Conv2d(1, n_filters, (window_size, embedding_dim)) 
        self.fc1 = nn.Linear(n_filters, num_labels)
        self.init_weights()
        
    def forward(self, x):
        # Pass the input through your layers in order
        out = self.embed(x)
        out = self.dropout(out)
        out = out.unsqueeze(1)
        out = self.conv1(out).squeeze(3)
        out = F.relu(out)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.fc1(self.dropout2(out))
        return out

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.fc1]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
window_size = 2
n_filters = 5
embedding_dim = 100
learning_rate = 0.1
dropout_prob = 0.0

def tuner(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, varying_param, param_range, finalRes):
    def calc(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, param_value, varying_param,  finalRes):
        model = TextCNN(input_size, embedding_dim, window_size, n_filters, num_labels, dropout_prob)
        model.cuda()
        # Loss and Optimizer
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        training_iter = data_iter(training_set, batch_size)
        train_eval_iter = eval_iter(training_set[0:500], batch_size)
        dev_iter = eval_iter(dev_set[0:500], batch_size)
        finalRes[varying_param][param_value]=training_loop(model, loss, optimizer, training_iter, dev_iter, train_eval_iter)
    for r in param_range:
        if varying_param=="dropout_prob":
            calc(input_size, embedding_dim, window_size, n_filters, num_labels,learning_rate, r, r, varying_param, finalRes)
        elif varying_param=="embedding_dim":
            calc(input_size, r, window_size, n_filters, num_labels,learning_rate, dropout_prob, r, varying_param, finalRes)
        elif varying_param=="learning_rate":
            calc(input_size, embedding_dim, window_size, n_filters, num_labels, r,dropout_prob, r, varying_param, finalRes)
        elif varying_param=="window_size":
            calc(input_size, embedding_dim, r, n_filters, num_labels,learning_rate, dropout_prob, r, varying_param, finalRes)
        elif varying_param=="n_filters":
            calc(input_size, embedding_dim, window_size, r, num_labels, learning_rate,dropout_prob, r, varying_param, finalRes)

finalcnn=collections.defaultdict(dict)
# print("Window size varying.")
# tuner(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, 'window_size', range(1,20, 2), finalcnn)
# print("n_filters varying.")
# tuner(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, 'n_filters', range(10,50, 10), finalcnn)
# print("embedding_dim varying.")
# tuner(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, 'embedding_dim', range(50,400, 25), finalcnn)
# print("Learning rate varying.")
# tuner(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, 'learning_rate', np.linspace(0.001, 0.1, num=10, endpoint=True), finalcnn)
print("Dropout prob varying.")
tuner(input_size, embedding_dim, window_size, n_filters, num_labels, learning_rate, dropout_prob, 'dropout_prob', np.linspace(0.1, 0.7, num=7, endpoint=True), finalcnn)

CNN_file="am8676_cnn"
fo=open(CNN_file, 'wb')
pickle.dump(finalcnn, fo)
fo.close()