#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import torchtext
import time
import os


# In[2]:


# Declare bi-directional LSTM for sentiment analyzer
class Network(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_outputs, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        # Bi-directional LSTM
        self.rnn = torch.nn.LSTM(emb_dim,
                                 hidden_size,
                                 num_layers=num_layers,
                                 dropout=0.3,
                                 bidirectional=True)
        # doubling size of hidden layers because of the bi-directional LSTM
        self.fc = torch.nn.Linear(hidden_size * 2, num_outputs)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        
    def forward(self, inputs):
        embs = self.embedding(inputs)
        output, _ = self.rnn(embs)
        output = self.fc(output[-1])
        return self.softmax(output)


# In[3]:


# Load data from tsv format file
def LoadTSV(file_path, columns, skip_header=True):
    return torchtext.data.TabularDataset(file_path, 'TSV', columns, skip_header=skip_header)


# In[4]:


# Data type for label(=target)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.long)
# Data type for phrases
TEXT = torchtext.data.Field(fix_length=50, use_vocab=True, lower=True)

train_columns = [
    ('PhraseId', None),
    ('SentenceId', None),
    ('Phrase', TEXT),
    ('Sentiment', LABEL)
]

test_columns = [
    ('PhraseId', None),
    ('SentenceId', None),
    ('Phrase', TEXT)
]

train = LoadTSV('./dataset/train.tsv/train.tsv', train_columns)
test = LoadTSV('./dataset/test.tsv/test.tsv', test_columns)
# Build vocab from phrases and use Glove vector for transfer learning
TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300), max_size=50000)
# Build vocab from labels
LABEL.build_vocab(train)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=64, device=device)
train_iter.repeat = False
test_iter.repeat = False


# In[5]:


vocab_size = len(TEXT.vocab)
emb_dim = 300
hidden_size = 300
num_outputs = 5
num_layers = 2

model = Network(vocab_size, emb_dim, hidden_size, num_outputs, num_layers=num_layers)
# Use Glove pretrained vector on the embedding layer
model.embedding.weight.data = TEXT.vocab.vectors
model.embedding.weight.require_grad = False

if torch.cuda.is_available():
    model = model.cuda()

epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

test_loss, test_accuracy = [], []

def training(model, dataset, optimizer, scheduler, epochs=10):
    model.train()
    training_loss, train_accuracy = [], []
    dataset_size = len(dataset.dataset)
    for epoch in range(epochs):
        epoch_begin = time.time()
        epoch_loss = 0.0
        epoch_corrects = 0
        print(f'------------- Epoch {epoch + 1} -------------')
        for batch in dataset:
            text, labels = batch.Phrase, batch.Sentiment
            if torch.cuda.is_available():
                text, labels = text.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(text)
            loss = F.nll_loss(output, labels, reduction='sum')
            _, preds = torch.max(output, dim=1)
            epoch_loss += loss.data.detach().item()
            epoch_corrects += preds.eq(labels.data.view_as(preds)).sum()
            
            loss.backward()
            optimizer.step()
        print(f'Loss / Accuracy : {epoch_loss / dataset_size :.4f} / {100. * epoch_corrects / dataset_size :.4f}% === {time.time() - epoch_begin}')
        scheduler.step()


# In[ ]:


training(model, train_iter, optimizer, scheduler, epochs=epochs)


# In[ ]:




