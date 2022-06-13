# -*- coding: utf-8 -*-
"""ECGAN_parsBERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pVXvvWwhNoz82-NVC-rw2r4x9gRNw7jh
"""

# !pip install transformers==4.3.2
import torch
import os
import io
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install sentencepiece

##Set random values
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

"""### Input Parameters

"""

dataSizeConstant = 0.1

#--------------------------------
#  Transformer parameters
#--------------------------------
max_seq_length = 128
batch_size = 32



#--------------------------------
#  GAN-BERT specific parameters
#--------------------------------
# number of hidden layers in the generator, 
# each of the size of the output space
num_hidden_layers_g = 1; 
# number of hidden layers in the discriminator, 
# each of the size of the input space
num_hidden_layers_d = 1; 

num_hidden_layers_c = 1;

# size of the generator's input noisy vectors
noise_size = 100
# dropout to be applied to discriminator's input vectors
out_dropout_rate = 0.2

# Replicate labeled data to balance poorly represented datasets, 
# e.g., less than 1% of labeled material
apply_balance = True

#--------------------------------
#  Optimization parameters
#--------------------------------
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5
learning_rate_classifier = 5e-5
epsilon = 1e-8
num_train_epochs = 100
multi_gpu = True
# Scheduler
apply_scheduler = False
warmup_proportion = 0.1
# Print
print_each_n_step = 10

#model_name = "HooshvareLab/bert-fa-base-uncased"
model_name = 'HooshvareLab/bert-fa-base-uncased' #@param ["HooshvareLab/bert-fa-base-uncased","HooshvareLab/bert-fa-zwnj-base"] {allow-input: true}

dataset = 'persiannews' #@param ["persiannews", "digikalamag"]
cmd_str = "git clone https://github.com/iamardian/{}.git".format(dataset)
os.system(cmd_str)
labeled_file = "./{}/train.csv".format(dataset)
unlabeled_file = "./{}/dev.csv".format(dataset)
test_filename = "./{}/test.csv".format(dataset)

print(labeled_file)
print(unlabeled_file)
print(test_filename)

"""Load the Tranformer Model"""

transformer = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""Function required to load the dataset"""

df = pd.read_csv(labeled_file, sep='\t')
lst_cnt = df.label_id.to_list()
label_list = list(set(lst_cnt))
print(label_list)

def get_qc_examples(input_file):
  """Creates examples for the training and dev sets."""
  
  df = pd.read_csv(input_file, sep='\t')
  lst_cnt = df.content.to_list()
  lst_id = df.label_id.to_list()
  examples = list(zip(lst_cnt,lst_id))
  return examples

"""Load the PersianNews and Digimag Datasets then separate labeled and unlabeled data."""

#Load the examples
labeled_examples = get_qc_examples(labeled_file)
subset = np.random.permutation([i for i in range(len(labeled_examples))])
number_of_sample = subset[:int(len(labeled_examples) * (dataSizeConstant))]
labeled_examples = [labeled_examples[i] for i in number_of_sample]

# unlabeled_examples = get_qc_examples(unlabeled_file)
unlabeled_examples = None
test_examples = get_qc_examples(test_filename)

"""Functions required to convert examples into Dataloader"""

def generate_data_loader(input_examples, label_masks, label_map, do_shuffle = False, balance_label_examples = False):
  '''
  Generate a Dataloader given the input examples, eventually masked if they are 
  to be considered NOT labeled.
  '''
  examples = []

  # Count the percentage of labeled examples  
  num_labeled_examples = 0
  for label_mask in label_masks:
    if label_mask: 
      num_labeled_examples += 1
  label_mask_rate = num_labeled_examples/len(input_examples)

  # if required it applies the balance
  for index, ex in enumerate(input_examples): 
    if label_mask_rate == 1 or not balance_label_examples:
      examples.append((ex, label_masks[index]))
    else:
      # IT SIMULATE A LABELED EXAMPLE
      if label_masks[index]:
        balance = int(1/label_mask_rate)
        balance = int(math.log(balance,2))
        if balance < 1:
          balance = 1
        for b in range(0, int(balance)):
          examples.append((ex, label_masks[index]))
      else:
        examples.append((ex, label_masks[index]))
  
  #-----------------------------------------------
  # Generate input examples to the Transformer
  #-----------------------------------------------
  input_ids = []
  input_mask_array = []
  label_mask_array = []
  label_id_array = []

  # Tokenization 
  for (text, label_mask) in examples:
    encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
    input_ids.append(encoded_sent)
    label_id_array.append(label_map[str(text[1])])
    label_mask_array.append(label_mask)
  
  # Attention to token (to ignore padded input wordpieces)
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]                          
    input_mask_array.append(att_mask)
  # Convertion to Tensor
  input_ids = torch.tensor(input_ids) 
  input_mask_array = torch.tensor(input_mask_array)
  label_id_array = torch.tensor(label_id_array, dtype=torch.long)
  label_mask_array = torch.tensor(label_mask_array)

  # Building the TensorDataset
  dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

  if do_shuffle:
    sampler = RandomSampler
  else:
    sampler = SequentialSampler

  # Building the DataLoader
  return DataLoader(
              dataset,  # The training samples.
              sampler = sampler(dataset), 
              batch_size = batch_size) # Trains with this batch size.

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

"""Convert the input examples into DataLoader"""

label_map = {}
for (i, label) in enumerate(label_list):
  label_map[str(label)] = i
#------------------------------
#   Load the train dataset
#------------------------------
train_examples = labeled_examples
#The labeled (train) dataset is assigned with a mask set to True
train_label_masks = np.ones(len(labeled_examples), dtype=bool)
#If unlabel examples are available
if unlabeled_examples:
  print("with unlabel examples")
  train_examples = train_examples + unlabeled_examples
  #The unlabeled (train) dataset is assigned with a mask set to False
  tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
  train_label_masks = np.concatenate([train_label_masks,tmp_masks])

train_dataloader = generate_data_loader(train_examples, train_label_masks, label_map, do_shuffle = True, balance_label_examples = apply_balance)

#------------------------------
#   Load the test dataset
#------------------------------
#The labeled (test) dataset is assigned with a mask set to True
test_label_masks = np.ones(len(test_examples), dtype=bool)

test_dataloader = generate_data_loader(test_examples, test_label_masks, label_map, do_shuffle = False, balance_label_examples = False)

"""We define the Generator and Discriminator as discussed in https://www.aclweb.org/anthology/2020.acl-main.191/"""

#------------------------------
#   The Generator as in 
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep
#------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Discriminator(nn.Module):
  def __init__(self, input_size=512, hidden_sizes=[512], dropout_rate=0.1):
    super(Discriminator, self).__init__()
    self.input_dropout = nn.Dropout(p=dropout_rate)
    layers = []
    hidden_sizes = [input_size] + hidden_sizes
    for i in range(len(hidden_sizes)-1):
        layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

    self.layers = nn.Sequential(*layers) #per il flatten
    self.logit = nn.Linear(hidden_sizes[-1],1) # fake/real.
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_rep):
    input_rep = self.input_dropout(input_rep)
    last_rep = self.layers(input_rep)
    logits = self.logit(last_rep)
    output = self.sigmoid(logits)
    return output.view(-1, 1).squeeze(1)

class Classifier(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Classifier, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],num_labels) # fake/real.
        self.softmax = nn.Softmax()
    
    def forward(self, input_rep):
        # input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        output = self.softmax(logits)
        return output

"""We instantiate the Discriminator and Generator"""

# The config file is required to get the dimension of the vector produced by 
# the underlying transformer
config = AutoConfig.from_pretrained(model_name)
hidden_size = int(config.hidden_size)
# Define the number and width of hidden layers
hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]
hidden_levels_c = [hidden_size for i in range(0, num_hidden_layers_c)]

#-------------------------------------------------
#   Instantiate the Generator and Discriminator
#-------------------------------------------------
generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, dropout_rate=out_dropout_rate)
classifier = Classifier(input_size=hidden_size, hidden_sizes=hidden_levels_c, num_labels=len(label_list), dropout_rate=out_dropout_rate)

# Put everything in the GPU if available
if torch.cuda.is_available():    
  generator.cuda()
  discriminator.cuda()
  classifier.cuda()
  transformer.cuda()
  if multi_gpu:
    transformer = torch.nn.DataParallel(transformer)

# print(config)

"""GAP"""

# data for plotting purposes
generatorLosses = []
discriminatorLosses = []
classifierLosses = []

#models parameters
transformer_vars = [i for i in transformer.parameters()]
d_vars = [v for v in discriminator.parameters()]
c_vars = transformer_vars + [v for v in classifier.parameters()]
# c_vars = [v for v in classifier.parameters()]
g_vars = [v for v in generator.parameters()]

#optimizer
dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
cfr_optimizer = torch.optim.AdamW(c_vars, lr=learning_rate_classifier)
gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator) 

advWeight = 0.1 # adversarial weight

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

total_acc = []
def print_accuurracy(index , acc):
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    title = ["epoch","acc"]
    total_acc.append([index , acc])
    tdf = pd.DataFrame(total_acc,columns=title)
    print (tdf)

def train(datasetloader):
  for epoch_i in range(0,num_train_epochs):

    classifier.train()

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    
    running_loss = 0.0
    total_train = 0
    correct_train = 0
    for i,batch in enumerate(datasetloader,0):

      # Progress update every print_each_n_step batches.
      if i % print_each_n_step == 0 and not i == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, len(datasetloader), elapsed))

      # Unpack this training batch from our dataloader. 
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)
      b_label_mask = batch[3].to(device)

      tmpBatchSize = b_input_ids.shape[0]

      # create label arrays
      true_label = torch.ones(tmpBatchSize,device=device)
      fake_label = torch.zeros(tmpBatchSize,device=device)

      noise = torch.zeros(tmpBatchSize, noise_size, device=device).uniform_(0, 1)
      fakeImageBatch = generator(noise)

      real_cpu = batch[0].to(device)
      batch_size = real_cpu.size(0)


      # Encode real data in the Transformer
      model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
      hidden_states = model_outputs[-1]

      # train discriminator on real images
      predictionsReal = discriminator(hidden_states)
      lossDiscriminator = loss(predictionsReal, true_label) #labels = 1
      lossDiscriminator.backward(retain_graph = True)

      # train discriminator on fake images
      predictionsFake = discriminator(fakeImageBatch)
      lossFake = loss(predictionsFake, fake_label) #labels = 0
      lossFake.backward(retain_graph= True)
      dis_optimizer.step() # update discriminator parameters

      # train generator 
      gen_optimizer.zero_grad()
      predictionsFake = discriminator(fakeImageBatch)
      lossGenerator = loss(predictionsFake, true_label) #labels = 1
      lossGenerator.backward(retain_graph = True)
      gen_optimizer.step()

      torch.autograd.set_detect_anomaly(True)
      fakeImageBatch = fakeImageBatch.detach().clone()

      # train classifier on real data
      predictions = classifier(hidden_states)
      realClassifierLoss = criterion(predictions, b_labels)
      realClassifierLoss.backward(retain_graph = True)

      cfr_optimizer.step()
      cfr_optimizer.zero_grad()

      # update the classifer on fake data
      predictionsFake = classifier(fakeImageBatch)
      # get a tensor of the labels that are most likely according to model
      predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
      confidenceThresh = .2

      # psuedo labeling threshold
      probs = F.softmax(predictionsFake, dim=1)
      mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
      toKeep = mostLikelyProbs > confidenceThresh
      if sum(toKeep) != 0:
          fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
          fakeClassifierLoss.backward()
      
      cfr_optimizer.step()

      # reset the gradients
      dis_optimizer.zero_grad()
      gen_optimizer.zero_grad()
      cfr_optimizer.zero_grad()

      # save losses for graphing
      generatorLosses.append(lossGenerator.item())
      discriminatorLosses.append(lossDiscriminator.item())
      classifierLosses.append(realClassifierLoss.item())

      # get train accurcy 
      if((i+1) % 10 == 0 ):
        classifier.eval()
        # accuracy
        _, predicted = torch.max(predictions, 1)
        total_train += b_labels.size(0)
        correct_train += predicted.eq(b_labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        print("({}) train_accuracy : {}".format(i+1,train_accuracy))
        classifier.train()
    
    print("Epoch " + str(epoch_i+1) + " Complete")
    validate(epoch_i)

def validate(epoch):
  classifier.eval()

  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_dataloader:

      b_input_ids = data[0].to(device)
      b_input_mask = data[1].to(device)
      b_labels = data[2].to(device)

      # inputs, labels = data
      # inputs, labels = data[0].to(device), data[1].to(device)


      model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
      hidden_states = model_outputs[-1]

      outputs = classifier(hidden_states)
      _, predicted = torch.max(outputs.data, 1)
      total += b_labels.size(0)
      correct += (predicted == b_labels).sum().item()

  accuracy = (correct / total) * 100 
  print_accuurracy(epoch+1,accuracy)
  print("{} / {} * 100 = {} ".format(correct,total,accuracy))
  classifier.train()

train(train_dataloader)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  title = ["epoch","acc"]
  tdf = pd.DataFrame(total_acc,columns=title)
  print (tdf)

