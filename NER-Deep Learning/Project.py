#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[ ]:





# In[8]:


get_ipython().system('pip install torch==1.8.0 torchtext==0.9.0')


# In[16]:


get_ipython().system('pip install --upgrade torch')


# In[83]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(1)

# Define hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.33
OUTPUT_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.1
EPOCHS = 50

# Define the model
class BLSTM(nn.Module):

    def __init__(self, vocab_size, tagset_size):
        super(BLSTM, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.hidden_dim = HIDDEN_DIM
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT
        self.output_dim = OUTPUT_DIM
        
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.blstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                             num_layers=self.num_layers, bidirectional=True,
                             dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.elu = nn.ELU()
        self.tag_projection = nn.Linear(self.output_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.blstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        elu_out = self.elu(tag_space)
        tag_scores = self.tag_projection(elu_out)
        return tag_scores
    
def train(model, optimizer, loss_function, train_data, dev_data):
    for epoch in range(EPOCHS):
        train_loss = 0.0
        num_correct = 0
        num_total = 0
        model.train()
        for sentence, tags in train_data:
            model.zero_grad()
            sentence = autograd.Variable(torch.LongTensor(sentence))
            targets = autograd.Variable(torch.LongTensor(tags))
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted_tags = torch.max(tag_scores, 1)
            num_correct += (predicted_tags == targets).sum().item()
            num_total += len(targets)

        precision, recall, f1 = evaluate(model, dev_data)
        print("Epoch {}: Train Loss: {:.5f}, Precision: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}" .format(epoch+1, train_loss, precision, recall, f1))

torch.save(model.state_dict(), 'blstm1.pt')


def evaluate(model, data):
    num_correct = 0
    num_total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for sentence, tags in data:
            sentence = autograd.Variable(torch.LongTensor(sentence))
            targets = autograd.Variable(torch.LongTensor(tags))
            tag_scores = model(sentence)
            _, predicted_tags = torch.max(tag_scores, 1)
            num_correct += (predicted_tags == targets).sum().item()
            num_total += len(targets)
            correct += (predicted_tags == targets).sum().item() * 1.0 / len(targets)

    precision = correct / num_correct
    recall = correct / num_total
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

# Read in the train and dev data
def read_data(file_path, has_tags=True):
    sentences = []
    tags = []
    with open(file_path) as f:
        sentence = []
        tags_sequence = []
        for line in f:
            if line == '\n':
                sentences.append(sentence)
                tags.append(tags_sequence)
                sentence = []
                tags_sequence = []
            else:
                line = line.strip().split()
                word = line[1]
                sentence.append(word)
                if has_tags:
                    tag = line[2]
                    tags_sequence.append(tag)
    return sentences, tags



# Define the paths to the train and dev data files
train_file_path = "train.txt"
dev_file_path = "dev.txt"
test_file_path = "test.txt"


# Read in the train and dev data
train_sentences, train_tags = read_data(train_file_path)
dev_sentences, dev_tags = read_data(dev_file_path)
test_sentences, _ = read_data(test_file_path, has_tags=False)

# Create the vocabulary and tag sets
word_to_ix = {}
tag_to_ix = {}
for sentence, tags in zip(train_sentences, train_tags):
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

# Add the unknown token to the vocabulary
word_to_ix["<UNK>"] = len(word_to_ix)

# Convert the sentences and tags to lists of indices
train_data = [(torch.tensor([word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]), 
               torch.tensor([tag_to_ix[tag] for tag in tags])) 
              for sentence, tags in zip(train_sentences, train_tags)]
dev_data = [(torch.tensor([word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]), 
             torch.tensor([tag_to_ix[tag] for tag in tags])) 
            for sentence, tags in zip(dev_sentences, dev_tags)]

test_data = [(torch.tensor([word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]), None) 
            for sentence in test_sentences]
# Create the model
model = BLSTM(len(word_to_ix), len(tag_to_ix))

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# # Train the model
# train(model, optimizer, loss_function, train_data, dev_data)

# def predict(model, data, word_to_ix, tag_to_ix, file_path):
#     with open(file_path, "w") as f:
#         model.eval()
#         with torch.no_grad():
#             for i, (sentence, tags) in enumerate(data):
#                 sentence = autograd.Variable(sentence.view(-1, 1))
#                 tag_scores = model(sentence)
#                 _, predicted_tags = torch.max(tag_scores, 1)
#                 for j, word_index in enumerate(sentence):
#                     word = list(word_to_ix.keys())[list(word_to_ix.values()).index(word_index.item())]
#                     gold_tag = ''
#                     if tags is not None:
#                         gold_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tags[j].item())]
#                     pred_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tags[j].item())]
#                     f.write(str(j+1) + " " + word + " " + gold_tag + " " + pred_tag + "\n")
#                 f.write("\n")

def predict(model, original_sentences, data, word_to_ix, tag_to_ix, file_path, include_gold_tag=True):
    with open(file_path, "w") as f:
        model.eval()
        with torch.no_grad():
            for i, (sentence, tags) in enumerate(data):
                original_sentence = original_sentences[i]
                sentence = autograd.Variable(sentence.view(-1, 1))
                tag_scores = model(sentence)
                _, predicted_tags = torch.max(tag_scores, 1)
                for j, word in enumerate(original_sentence):
                    gold_tag = ''
                    if tags is not None and include_gold_tag:
                        gold_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tags[j].item())]
                    pred_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tags[j].item())]
                    f.write(str(j+1) + " " + word + " " + gold_tag + " " + pred_tag + "\n")
                f.write("\n")


if pred_file_path is not None:
    with open(pred_file_path, "a") as f_pred:
        for i, (sentence, tags) in enumerate(dev_data):
            sentence = autograd.Variable(sentence.view(-1, 1))
            tag_scores = model(sentence)
            _, predicted_tags = torch.max(tag_scores, 1)
            for j, word_index in enumerate(sentence):
                word = list(word_to_ix.keys())[list(word_to_ix.values()).index(word_index.item())]
                gold_tag = ''
                if tags is not None:
                    gold_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tags[j].item())]
                pred_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tags[j].item())]
                f_pred.write(str(j+1) + " " + word + " " + " " + pred_tag + "\n")
            f_pred.write("\n")
                
pred_file_path = "dev1.out"
perl_dev_file_path = "perl_dev1.out"
test_pred_file_path = "test1.out"

# Generate the predictions and write them to the prediction files
predict(model, dev_sentences, dev_data, word_to_ix, tag_to_ix, pred_file_path, include_gold_tag=False)
predict(model, dev_sentences, dev_data, word_to_ix, tag_to_ix, perl_dev_file_path, include_gold_tag=True)
predict(model, test_sentences, test_data, word_to_ix, tag_to_ix, test_pred_file_path, include_gold_tag=False)


# # Task2

# In[84]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import gzip
# Set random seed for reproducibility
torch.manual_seed(1)

# Define hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.33
OUTPUT_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.1
EPOCHS = 10





def load_glove_embeddings(glove_path, word_to_ix, embed_dim):
    embeddings = {}
    with gzip.open(glove_path, 'rt', encoding='utf-8') as f:  # Use gzip.open instead of open
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            embeddings[word] = vector

    embedding_matrix = torch.zeros(len(word_to_ix), embed_dim)
    for word, index in word_to_ix.items():
        if word in embeddings:
            embedding_matrix[index] = torch.FloatTensor(embeddings[word])
        else:
            embedding_matrix[index] = torch.FloatTensor(embed_dim).uniform_(-0.5, 0.5)

    return embedding_matrix



# Define the model
class BLSTM(nn.Module):

    def __init__(self, vocab_size, tagset_size, pretrained_embeddings):
        super(BLSTM, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.hidden_dim = HIDDEN_DIM
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT
        self.output_dim = OUTPUT_DIM
        
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.blstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                             num_layers=self.num_layers, bidirectional=True,
                             dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.elu = nn.ELU()
        self.tag_projection = nn.Linear(self.output_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.blstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        elu_out = self.elu(tag_space)
        tag_scores = self.tag_projection(elu_out)
        return tag_scores
    
# Define the training function
def train(model, optimizer, loss_function, train_data, dev_data):
    for epoch in range(EPOCHS):
        train_loss = 0.0
        num_correct = 0
        num_total = 0
        model.train()
        for sentence, tags in train_data:
            model.zero_grad()
            sentence = autograd.Variable(torch.LongTensor(sentence))
            targets = autograd.Variable(torch.LongTensor(tags))
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted_tags = torch.max(tag_scores, 1)
            num_correct += (predicted_tags == targets).sum().item()
            num_total += len(targets)

        precision, recall, f1 = evaluate(model, dev_data)
        print("Epoch {}: Train Loss: {:.5f}, Precision: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}" .format(epoch+1, train_loss, precision, recall, f1))

torch.save(model.state_dict(), 'blstm2.pt')

def evaluate(model, data):
    num_correct = 0
    num_total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for sentence, tags in data:
            sentence = autograd.Variable(torch.LongTensor(sentence))
            targets = autograd.Variable(torch.LongTensor(tags))
            tag_scores = model(sentence)
            _, predicted_tags = torch.max(tag_scores, 1)
            num_correct += (predicted_tags == targets).sum().item()
            num_total += len(targets)
            correct += (predicted_tags == targets).sum().item() * 1.0 / len(targets)

    precision = correct / num_correct
    recall = correct / num_total
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

# Read in the train and dev data
def read_data(file_path, has_tags=True):
    sentences = []
    tags = []
    with open(file_path) as f:
        sentence = []
        tags_sequence = []
        for line in f:
            if line == '\n':
                sentences.append(sentence)
                tags.append(tags_sequence)
                sentence = []
                tags_sequence = []
            else:
                line = line.strip().split()
                word = line[1]
                sentence.append(word)
                if has_tags:
                    tag = line[2]
                    tags_sequence.append(tag)
    return sentences, tags


glove_path = "glove.6B.100d.gz"
pretrained_embeddings = load_glove_embeddings(glove_path, word_to_ix, EMBEDDING_DIM)

# Define the paths to the train and dev data files
train_file_path = "train.txt"
dev_file_path = "dev.txt"
test_file_path = "test.txt"


# Read in the train and dev data
train_sentences, train_tags = read_data(train_file_path)
dev_sentences, dev_tags = read_data(dev_file_path)
test_sentences, _ = read_data(test_file_path, has_tags=False)

# Create the vocabulary and tag sets
word_to_ix = {}
tag_to_ix = {}
for sentence, tags in zip(train_sentences, train_tags):
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

# Add the unknown token to the vocabulary
word_to_ix["<UNK>"] = len(word_to_ix)

# Convert the sentences and tags to lists of indices
train_data = [(torch.tensor([word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]), 
               torch.tensor([tag_to_ix[tag] for tag in tags])) 
              for sentence, tags in zip(train_sentences, train_tags)]
dev_data = [(torch.tensor([word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]), 
             torch.tensor([tag_to_ix[tag] for tag in tags])) 
            for sentence, tags in zip(dev_sentences, dev_tags)]

test_data = [(torch.tensor([word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]), None) 
            for sentence in test_sentences]
# Create the model
model = BLSTM(len(word_to_ix), len(tag_to_ix), pretrained_embeddings)

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model
train(model, optimizer, loss_function, train_data, dev_data)

# def predict(model, data, word_to_ix, tag_to_ix, file_path):
#     with open(file_path, "w") as f:
#         model.eval()
#         with torch.no_grad():
#             for i, (sentence, tags) in enumerate(data):
#                 sentence = autograd.Variable(sentence.view(-1, 1))
#                 tag_scores = model(sentence)
#                 _, predicted_tags = torch.max(tag_scores, 1)
#                 for j, word_index in enumerate(sentence):
#                     word = list(word_to_ix.keys())[list(word_to_ix.values()).index(word_index.item())]
#                     gold_tag = ''
#                     if tags is not None:
#                         gold_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tags[j].item())]
#                     pred_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tags[j].item())]
#                     f.write(str(j+1) + " " + word + " " + gold_tag + " " + pred_tag + "\n")
#                 f.write("\n")

def predict(model, original_sentences, data, word_to_ix, tag_to_ix, file_path, include_gold_tag=True):
    with open(file_path, "w") as f:
        model.eval()
        with torch.no_grad():
            for i, (sentence, tags) in enumerate(data):
                original_sentence = original_sentences[i]
                sentence = autograd.Variable(sentence.view(-1, 1))
                tag_scores = model(sentence)
                _, predicted_tags = torch.max(tag_scores, 1)
                for j, word in enumerate(original_sentence):
                    gold_tag = ''
                    if tags is not None and include_gold_tag:
                        gold_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tags[j].item())]
                    pred_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tags[j].item())]
                    f.write(str(j+1) + " " + word + " " + gold_tag + " " + pred_tag + "\n")
                f.write("\n")


if pred_file_path is not None:
    with open(pred_file_path, "a") as f_pred:
        for i, (sentence, tags) in enumerate(dev_data):
            sentence = autograd.Variable(sentence.view(-1, 1))
            tag_scores = model(sentence)
            _, predicted_tags = torch.max(tag_scores, 1)
            for j, word_index in enumerate(sentence):
                word = list(word_to_ix.keys())[list(word_to_ix.values()).index(word_index.item())]
                gold_tag = ''
                if tags is not None:
                    gold_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tags[j].item())]
                pred_tag = list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tags[j].item())]
                f_pred.write(str(j+1) + " " + word + " " + " " + pred_tag + "\n")
            f_pred.write("\n")
                
pred_file_path = "dev2.out"
perl_dev_file_path = "perl_dev2.out"
test_pred_file_path = "test2.out"

# Generate the predictions and write them to the prediction files
predict(model, dev_sentences, dev_data, word_to_ix, tag_to_ix, pred_file_path, include_gold_tag=False)
predict(model, dev_sentences, dev_data, word_to_ix, tag_to_ix, perl_dev_file_path, include_gold_tag=True)
predict(model, test_sentences, test_data, word_to_ix, tag_to_ix, test_pred_file_path, include_gold_tag=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:


import torch
from torch.autograd import Variable

# Create a new instance of the model with the same architecture
loaded_model = BLSTM(len(word_to_ix), len(tag_to_ix), pretrained_embeddings)

# Load the saved state dictionary into the new model
loaded_model.load_state_dict(torch.load('blstm2.pt'))

# Set the model to evaluation mode
loaded_model.eval()


# In[87]:


dev_pred_file_path = "dev_predictions.txt"
test_pred_file_path = "test_predictions.txt"

# Generate the predictions and write them to the prediction files
predict(loaded_model, dev_sentences, dev_data, word_to_ix, tag_to_ix, dev_pred_file_path)
predict(loaded_model, test_sentences, test_data, word_to_ix, tag_to_ix, test_pred_file_path)


# In[88]:


import torch
from torch.autograd import Variable

# Create a new instance of the model with the same architecture
loaded_model = BLSTM(len(word_to_ix), len(tag_to_ix))

# Load the saved state dictionary into the new model
loaded_model.load_state_dict(torch.load('blstm1.pt'))

# Set the model to evaluation mode
loaded_model.eval()



torch.save(model_lstm.state_dict(), 'blstm1.pt')
loaded_model_lstm = LSTM(vocab_size = len(vocab)+1, embedding_dim = 100, lstm_hidden_size = 256, fc_num_neurons = 128, num_classes = len(labels)).to(device)
loaded_model_lstm.load_state_dict(torch.load('blstm1.pt'))


# In[89]:


dev_pred_file_path = "dev_predictions1.txt"
test_pred_file_path = "test_predictions1.txt"

# Generate the predictions and write them to the prediction files
predict(loaded_model, dev_sentences, dev_data, word_to_ix, tag_to_ix, dev_pred_file_path)
predict(loaded_model, test_sentences, test_data, word_to_ix, tag_to_ix, test_pred_file_path)


# In[91]:


print(len(word_to_ix))


# In[ ]:




