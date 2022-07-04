
# !pip install transformers==4.3.2
from asyncio import constants
import sys
import getopt
from genericpath import exists
from sklearn.model_selection import train_test_split
import glob
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
import xlsxwriter
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install sentencepiece


def create_path_if_not_exists(dir_path):
    # print("call create_path_if_not_exists")
    # print(f"dir_path : {dir_path}")
    if os.path.exists(dir_path):
        return
    os.makedirs(dir_path, exist_ok=True)


def datasets_summary(train, validation, test, dataset_name):
    train_data = pd.DataFrame(train, columns=["data", "label_id"])
    validation_data = pd.DataFrame(validation, columns=["data", "label_id"])
    test_data = pd.DataFrame(test, columns=["data", "label_id"])

    train_info = train_data.groupby(["label_id"]).size()
    train_len = len(train_data)

    validation_info = validation_data.groupby(["label_id"]).size()
    validation_len = len(validation_data)

    test_info = test_data.groupby(["label_id"]).size()
    test_len = len(test_data)

    log_print()
    log_print(f"Dataset : {dataset_name}")
    log_print()
    log_print("Train Dataset")
    log_print("===============================")
    log_print(train_info.to_string())
    log_print(f"Total : {train_len}")
    log_print()
    log_print("Validation Dataset")
    log_print("===============================")
    log_print(validation_info.to_string())
    log_print(f"Total : {validation_len}")
    log_print()
    log_print("Test Dataset")
    log_print("===============================")
    log_print(test_info.to_string())
    log_print(f"Total : {test_len}")


def print_params(params):
    param_list = []
    cols = ["param", "value"]
    for x in params:
        param_list.append([x, params[x]])
    df = pd.DataFrame(param_list, columns=cols)
    log_print(df.to_string(index=False))


##########################
# GET PARAMS WITH SWITCH
##########################
argumentList = sys.argv[1:]
# Options
options = "hd:p:w:t:e:l:m:o:c:g:r:"
# Long options
long_options = ["help", "dataset", "percentage",
                "weight", "thresh", "epochs", "label_balance", "mode", "optimizer", "--classifier_rate", "--generator_rate", "--discriminator_rate"]

dataset_name = "persiannews"
percentage_labeled_data = 0.1
adversarial_weight = 0.1
confidence_thresh = 0.2
num_epochs = 10
balance_label = False
train_BERT_mode = 0  # 0 = freeze | -1 = full | positive number = n latest layer of BERT
optimizer = "adamW"
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5
learning_rate_classifier = 5e-5
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
    # checking each argument
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("Displaying Help")
        elif currentArgument in ("-d", "--dataset"):
            dataset_name = currentValue
        elif currentArgument in ("-p", "--percentage"):
            percentage_labeled_data = float(currentValue)
        elif currentArgument in ("-w", "--weight"):
            adversarial_weight = float(currentValue)
        elif currentArgument in ("-t", "--thresh"):
            confidence_thresh = float(currentValue)
        elif currentArgument in ("-e", "--epochs"):
            num_epochs = int(currentValue)
        elif currentArgument in ("-l", "--label_balance"):
            balance_label = bool(currentValue)
        elif currentArgument in ("-m", "--mode"):
            train_BERT_mode = int(currentValue)
        elif currentArgument in ("-o", "--optimizer"):
            optimizer = str(currentValue)
        elif currentArgument in ("-c", "--classifier_rate"):
            learning_rate_classifier = float(currentValue)
        elif currentArgument in ("-g", "--generator_rate"):
            learning_rate_generator = float(currentValue)
        elif currentArgument in ("-r", "--discriminator_rate"):
            learning_rate_discriminator = float(currentValue)
except getopt.error as err:
    # output error, and return with an error code
    print(str(err))

# print("Dataset : {} \nPercentage : {}".format(
#     dataset_name, percentage_labeled_data))


default_path_str = "/content/drive/MyDrive/NLP/save/"
dir_name = f"ec_gan|" +\
    f"{dataset_name}|" +\
    f"{percentage_labeled_data}|" +\
    f"{adversarial_weight}|" +\
    f"{confidence_thresh}|" +\
    f"{num_epochs}|" +\
    f"{balance_label}|" +\
    f"{train_BERT_mode}|" +\
    f"{optimizer}|" +\
    f"d_{learning_rate_discriminator}|" +\
    f"g_{learning_rate_generator}|" +\
    f"c_{learning_rate_classifier}"

models_path = os.path.join(default_path_str, dir_name)
best_model_name = "best_model"

create_path_if_not_exists(models_path)


def log_print(*args):
    print()
    for a in args:
        print(a, end=' ')
        print(a, file=open(models_path+"/log_file.txt", 'a'), end=' ')


# Set random values
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
    # print('There are %d GPU(s) available.' % torch.cuda.device_count())
    # print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    # print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

"""### Input Parameters

"""


dataSizeConstant = percentage_labeled_data

# --------------------------------
#  Transformer parameters
# --------------------------------
max_seq_length = 128
batch_size = 32


# --------------------------------
#  GAN-BERT specific parameters
# --------------------------------
# number of hidden layers in the generator,
# each of the size of the output space
num_hidden_layers_g = 1
# number of hidden layers in the discriminator,
# each of the size of the input space
num_hidden_layers_d = 1

num_hidden_layers_c = 1

# size of the generator's input noisy vectors
noise_size = 100
# dropout to be applied to discriminator's input vectors
out_dropout_rate = 0.2

# Replicate labeled data to balance poorly represented datasets,
# e.g., less than 1% of labeled material
apply_balance = True

# --------------------------------
#  Optimization parameters
# --------------------------------

epsilon = 1e-8

num_train_epochs = num_epochs

multi_gpu = False
# Scheduler
apply_scheduler = False
warmup_proportion = 0.1
# Print
print_each_n_step = 10

#model_name = "HooshvareLab/bert-fa-base-uncased"
# @param ["HooshvareLab/bert-fa-base-uncased","HooshvareLab/bert-fa-zwnj-base"] {allow-input: true}
model_name = 'HooshvareLab/bert-fa-base-uncased'

dataset = dataset_name
cmd_str = "git clone https://github.com/iamardian/{}.git".format(dataset)
os.system(cmd_str)
labeled_file = "./{}/train.csv".format(dataset)
validation_file = "./{}/dev.csv".format(dataset)
test_filename = "./{}/test.csv".format(dataset)

# print(labeled_file)
# print(validation_file)
# print(test_filename)

"""Load the Tranformer Model"""

transformer = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""Function required to load the dataset"""

df = pd.read_csv(labeled_file, sep='\t')
lst_cnt = df.label_id.to_list()
label_list = list(set(lst_cnt))
# print(label_list)


def get_balance_labeled_example(file_path, size_of_data):
    df = pd.read_csv(file_path, sep='\t')
    x = df.content.to_list()
    y = df.label_id.to_list()
    X_train, _, y_train, _ = train_test_split(
        x, y, train_size=size_of_data, random_state=42, stratify=y)
    return list(zip(X_train, y_train))


def get_qc_examples(input_file, dataSizeConstant, title):
    """Creates examples for the training and dev sets."""

    df = pd.read_csv(input_file, sep='\t')
    lst_cnt = df.content.to_list()
    lst_id = df.label_id.to_list()
    examples = list(zip(lst_cnt, lst_id))
    if dataSizeConstant > 0:
        subset = np.random.permutation([i for i in range(len(examples))])
        number_of_sample = subset[:int(len(examples) * (dataSizeConstant))]
        examples = [examples[i] for i in number_of_sample]
    return examples


# Load the examples
if balance_label:
    labeled_examples = get_balance_labeled_example(
        labeled_file, dataSizeConstant)
else:
    labeled_examples = get_qc_examples(
        labeled_file, dataSizeConstant, "Train Dataset")
    # subset = np.random.permutation([i for i in range(len(labeled_examples))])
    # number_of_sample = subset[:int(len(labeled_examples) * (dataSizeConstant))]
    # labeled_examples = [labeled_examples[i] for i in number_of_sample]


validation_examples = get_qc_examples(
    validation_file, -1, "Validation Dataset")
test_examples = get_qc_examples(test_filename, -1, "Test Dataset")

datasets_summary(labeled_examples, validation_examples,
                 test_examples, dataset_name)

"""Functions required to convert examples into Dataloader"""


def generate_data_loader(input_examples, label_masks, label_map, do_shuffle=False, balance_label_examples=False):
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
                balance = int(math.log(balance, 2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    examples.append((ex, label_masks[index]))
            else:
                examples.append((ex, label_masks[index]))

    # -----------------------------------------------
    # Generate input examples to the Transformer
    # -----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    # Tokenization
    for (text, label_mask) in examples:
        encoded_sent = tokenizer.encode(
            text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
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
    dataset = TensorDataset(input_ids, input_mask_array,
                            label_id_array, label_mask_array)

    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
        dataset,  # The training samples.
        sampler=sampler(dataset),
        batch_size=batch_size)  # Trains with this batch size.


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
# ------------------------------
#   Load the train dataset
# ------------------------------
train_examples = labeled_examples
# The labeled (train) dataset is assigned with a mask set to True
train_label_masks = np.ones(len(labeled_examples), dtype=bool)
train_dataloader = generate_data_loader(
    train_examples, train_label_masks, label_map, do_shuffle=True, balance_label_examples=apply_balance)


validation_label_masks = np.ones(len(validation_examples), dtype=bool)
validation_dataloader = generate_data_loader(
    validation_examples, validation_label_masks, label_map, do_shuffle=True, balance_label_examples=apply_balance)


# ------------------------------
#   Load the test dataset
# ------------------------------
# The labeled (test) dataset is assigned with a mask set to True
test_label_masks = np.ones(len(test_examples), dtype=bool)
test_dataloader = generate_data_loader(
    test_examples, test_label_masks, label_map, do_shuffle=False, balance_label_examples=False)

"""We define the Generator and Discriminator as discussed in https://www.aclweb.org/anthology/2020.acl-main.191/"""

# ------------------------------
#   The Generator as in
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
# ------------------------------


class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                          nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep
# ------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
# ------------------------------


class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                          nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers)  # per il flatten
        self.logit = nn.Linear(hidden_sizes[-1], 1)  # fake/real.
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
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                          nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers)  # per il flatten
        self.logit = nn.Linear(hidden_sizes[-1], num_labels)  # fake/real.
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

# -------------------------------------------------
#   Instantiate the Generator and Discriminator
# -------------------------------------------------
generator = Generator(noise_size=noise_size, output_size=hidden_size,
                      hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
discriminator = Discriminator(
    input_size=hidden_size, hidden_sizes=hidden_levels_d, dropout_rate=out_dropout_rate)
classifier = Classifier(input_size=hidden_size, hidden_sizes=hidden_levels_c,
                        num_labels=len(label_list), dropout_rate=out_dropout_rate)

# Put everything in the GPU if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    transformer.cuda()
    if multi_gpu:
        transformer = torch.nn.DataParallel(transformer)


# data for plotting purposes
generatorLosses = []
discriminatorLosses = []
classifierLosses = []


def print_model_params(model):
    for item in model.parameters():
        log_print(item.requires_grad)


def bert_params_for_tune(model, mode):
    if mode < 0:
        return [x for x in model.parameters()]

    for param in model.embeddings.parameters():
        param.requires_grad = False

    layers = model.encoder.layer
    for i, layer in enumerate(layers):
        if i < len(layers) - mode:
            params = layer.parameters()
            for param in params:
                param.requires_grad = False
        else:
            break

    return [x for x in model.parameters()]


# models parameters
# transformer_params = [x for x in transformer.parameters()]
# print("========================== BEFORE ==========================")
# print_model_params(transformer)

transformer_vars = bert_params_for_tune(transformer, train_BERT_mode)

# print("========================== AFTER ==========================")
# print_model_params(transformer)
# print("========================== END ==========================")


d_vars = [v for v in discriminator.parameters()]
c_vars = transformer_vars + [v for v in classifier.parameters()]
# c_vars = [v for v in classifier.parameters()]
g_vars = [v for v in generator.parameters()]


def optimizer_adam():
    # optimizer
    dis_optimizer = torch.optim.Adam(d_vars, lr=learning_rate_discriminator)
    cfr_optimizer = torch.optim.Adam(c_vars, lr=learning_rate_classifier)
    gen_optimizer = torch.optim.Adam(g_vars, lr=learning_rate_generator)
    return dis_optimizer, cfr_optimizer, gen_optimizer


def optimizer_adamW():
    # optimizer
    dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
    cfr_optimizer = torch.optim.AdamW(c_vars, lr=learning_rate_classifier)
    gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator)
    return dis_optimizer, cfr_optimizer, gen_optimizer


optimizations = {
    "adam": optimizer_adam,
    "adamW": optimizer_adamW
}

dis_optimizer, cfr_optimizer, gen_optimizer = optimizations[optimizer]()

# # optimizer
# dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
# cfr_optimizer = torch.optim.AdamW(c_vars, lr=learning_rate_classifier)
# gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator)

advWeight = adversarial_weight  # adversarial weight
offset = -1

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

total_acc_validation = []
total_acc_evaluation = []
total_acc_test = []

total_label_base_accuracy_validation = []
total_label_base_accuracy_evaluation = []
total_label_base_accuracy_test = []

best_model_accuracy = 0


params_obj = {
    "Dataset": dataset_name,
    "Percentage": percentage_labeled_data,
    "Adversarial Weight": adversarial_weight,
    "Epochs": num_epochs,
    "Balance label": balance_label,
    "Batch size": batch_size,
    "Model name": model_name,
    "train_BERT_mode": train_BERT_mode,
    "optimizer": optimizer,
    "learning_rate_discriminator": learning_rate_discriminator,
    "learning_rate_generator": learning_rate_generator,
    "learning_rate_classifier": learning_rate_classifier,
}
log_print("===========================================")
print_params(params_obj)
log_print("===========================================")


def load_best_model(load_path):
    # print("call load_best_model")
    output_dir = load_path+"/best"
    if not os.path.exists(load_path):
        # print("not exists path : ", load_path)
        return False
    if not os.path.exists(output_dir):
        # print("not exists path : ", output_dir)
        return False
    best_model_path = load_path + "/" + f"{best_model_name}.pth"
    checkpoint = torch.load(best_model_path)
    # transformer = AutoModel.from_pretrained(model_name)
    # classifier = Classifier(input_size=hidden_size, hidden_sizes=hidden_levels_c,
    #                         num_labels=len(label_list), dropout_rate=out_dropout_rate)
    # transformer.load_state_dict(checkpoint['transformer_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    # tokenizer2 = tokenizer.from_pretrained(output_dir)
    transformer2 = AutoModel.from_pretrained(output_dir)
    return transformer2, classifier


def save_best_model(save_path, epoch, accuracy):
    # print("call save_best_model")
    output_dir = save_path+"/best"
    # print(f"output_dir : {output_dir}")
    create_path_if_not_exists(output_dir)
    create_path_if_not_exists(save_path)
    global best_model_accuracy
    if best_model_accuracy >= accuracy:
        log_print(f"best_model_accuracy : {best_model_accuracy}")
        return
    best_model_accuracy = accuracy
    log_print(f"best_model_accuracy : {best_model_accuracy}")
    best_model_path = os.path.join(save_path, f"{best_model_name}.pth")
    model_to_save = transformer.module if hasattr(
        transformer, 'module') else transformer
    model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    torch.save({
        'epoch': epoch,
        # 'transformer_state_dict': transformer.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, best_model_path)
    # print("Best model Saved")


def print_validation_accuracy(index, acc):
    # print("call print_validation_accuracy")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # title = ["epoch", "acc"]
        # total_acc_validation.append([index, acc])
        total_acc_validation.append(acc)
        tdf = pd.DataFrame(total_acc_validation)
        # print("validation")
        # print(tdf)
        # print("validation")


def print_evaluation_accuracy(index, acc):
    # print("call print_evaluation_accuracy")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # title = ["epoch", "acc"]
        total_acc_evaluation.append(acc)
        tdf = pd.DataFrame(total_acc_evaluation)
        # print("evaluation")
        # print(tdf)
        # print("evaluation")


def print_test_accuracy(acc):
    # print("call print_test_accuracy")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        title = ["acc"]
        total_acc_test.append(acc)
        tdf = pd.DataFrame(total_acc_test, columns=title)
        # print("test")
        # print(tdf)
        # print("test")


def find_latest_model_name(dir_path):
    # print("call find_latest_model_name")
    model_name = ""
    if not os.path.exists(dir_path):
        return model_name
    files = sorted(filter(os.path.isfile, glob.glob(dir_path + '/*.pth')))
    if len(files) == 0:
        return model_name
    for f in reversed(range(len(files))):
        if "best" in files[f]:
            continue
        else:
            model_name = files[f]
            break
    return model_name


def load_params(load_path, classifier, generator, discriminator, transformer, cfr_optimizer, gen_optimizer, dis_optimizer):
    # print("call load_params")
    if not os.path.exists(load_path+"/best"):
        # print("not exists path : ", load_path)
        return
    model_path = find_latest_model_name(load_path)
    # print(f"latest model : {model_path}")
    if model_path == "":
        return
    # print(f"model_path : {model_path}")
    checkpoint = torch.load(model_path)
    global offset, best_model_accuracy, generatorLosses, discriminatorLosses, classifierLosses, total_acc_validation, total_acc_evaluation, total_label_base_accuracy_validation, total_label_base_accuracy_evaluation, total_label_base_accuracy_test
    offset = checkpoint['epoch']
    # print(f"offset : {offset}")
    best_model_accuracy = checkpoint['best_model_accuracy']

    transformer.load_state_dict(checkpoint['transformer_state_dict'])

    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    cfr_optimizer.load_state_dict(checkpoint['cfr_optimizer_state_dict'])

    generator.load_state_dict(checkpoint['generator_state_dict'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])

    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])

    generatorLosses = checkpoint['generatorLosses']
    discriminatorLosses = checkpoint['discriminatorLosses']
    classifierLosses = checkpoint['classifierLosses']

    total_acc_validation = checkpoint['total_acc_validation']
    total_acc_evaluation = checkpoint['total_acc_evaluation']
    total_label_base_accuracy_validation = checkpoint['total_label_base_accuracy_validation']
    total_label_base_accuracy_evaluation = checkpoint['total_label_base_accuracy_evaluation']
    total_label_base_accuracy_test = checkpoint['total_label_base_accuracy_test']


def save_params(epoch, save_path):
    # print("call save_params")
    try:
        create_path_if_not_exists(save_path)
        model_name_path = f'{str(epoch).zfill(3)}_{dataset_name}_{percentage_labeled_data}_{adversarial_weight}_{confidence_thresh}.pth'
        model_path_name = os.path.join(save_path, model_name_path)
        torch.save({
            'epoch': epoch,
            'best_model_accuracy': best_model_accuracy,

            'transformer_state_dict': transformer.state_dict(),

            'classifier_state_dict': classifier.state_dict(),
            'cfr_optimizer_state_dict': cfr_optimizer.state_dict(),

            'generator_state_dict': generator.state_dict(),
            'gen_optimizer_state_dict': gen_optimizer.state_dict(),

            'discriminator_state_dict': discriminator.state_dict(),
            'dis_optimizer_state_dict': dis_optimizer.state_dict(),

            'generatorLosses': generatorLosses,
            'discriminatorLosses': discriminatorLosses,
            'classifierLosses': classifierLosses,

            'total_acc_validation': total_acc_validation,
            'total_acc_evaluation': total_acc_evaluation,
            'total_label_base_accuracy_validation': total_label_base_accuracy_validation,
            'total_label_base_accuracy_evaluation': total_label_base_accuracy_evaluation,
            'total_label_base_accuracy_test': total_label_base_accuracy_test,
        }, model_path_name)
        remove_previous_models(save_path, model_name_path)
    except:
        log_print("save model failed ...")


def write_to_file(file_path):
    with open(file_path, 'wb') as f:  # binary because we need count bytes
        max_size = 1 * 1024  # I assume num2 in kb
        message = "hello world"
        msg_bytes = message.encode('utf-8')
        bytes_written = 0
        while bytes_written < max_size:  # if you dont need breaking the last phrase
            f.write(msg_bytes)
            bytes_written += len(msg_bytes)


def remove_previous_models(dir_path, current_model):
    filelist = sorted(filter(os.path.isfile, glob.glob(dir_path + '/*')))
    # print(filelist)
    for f in filelist:
        if (not ("best" in f) and not (current_model in f)):
            write_to_file(f)
            os.remove(os.path.join(dir_path, f))


def train(datasetloader):
    log_print("Training Start : ")
    load_params(models_path, classifier, generator, discriminator,
                transformer, cfr_optimizer, gen_optimizer, dis_optimizer)
    for epoch_i in range(offset+1, num_train_epochs):

        classifier.train()

        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        log_print("")
        log_print(
            '======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
        log_print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, batch in enumerate(datasetloader, 0):

            # Progress update every print_each_n_step batches.
            if i % print_each_n_step == 0 and not i == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                log_print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    i, len(datasetloader), elapsed))

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_label_mask = batch[3].to(device)

            tmpBatchSize = b_input_ids.shape[0]

            # create label arrays
            true_label = torch.ones(tmpBatchSize, device=device)
            fake_label = torch.zeros(tmpBatchSize, device=device)

            noise = torch.zeros(tmpBatchSize, noise_size,
                                device=device).uniform_(0, 1)
            fakeImageBatch = generator(noise)

            real_cpu = batch[0].to(device)
            batch_size = real_cpu.size(0)

            # Encode real data in the Transformer
            model_outputs = transformer(
                b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            # train discriminator on real images
            predictionsReal = discriminator(hidden_states)
            lossDiscriminator = loss(predictionsReal, true_label)  # labels = 1
            lossDiscriminator.backward(retain_graph=True)

            # train discriminator on fake images
            predictionsFake = discriminator(fakeImageBatch)
            lossFake = loss(predictionsFake, fake_label)  # labels = 0
            lossFake.backward(retain_graph=True)
            dis_optimizer.step()  # update discriminator parameters

            # train generator
            gen_optimizer.zero_grad()
            predictionsFake = discriminator(fakeImageBatch)
            lossGenerator = loss(predictionsFake, true_label)  # labels = 1
            lossGenerator.backward(retain_graph=True)
            gen_optimizer.step()

            torch.autograd.set_detect_anomaly(True)
            fakeImageBatch = fakeImageBatch.detach().clone()

            # train classifier on real data
            predictions = classifier(hidden_states)
            realClassifierLoss = criterion(predictions, b_labels)
            realClassifierLoss.backward(retain_graph=True)

            cfr_optimizer.step()
            cfr_optimizer.zero_grad()

            # update the classifer on fake data
            predictionsFake = classifier(fakeImageBatch)
            # get a tensor of the labels that are most likely according to model
            predictedLabels = torch.argmax(
                predictionsFake, 1)  # -> [0 , 5, 9, 3, ...]
            confidenceThresh = confidence_thresh

            # psuedo labeling threshold
            probs = F.softmax(predictionsFake, dim=1)
            mostLikelyProbs = np.asarray(
                [probs[i, predictedLabels[i]].item() for i in range(len(probs))])
            toKeep = mostLikelyProbs > confidenceThresh
            if sum(toKeep) != 0:
                fakeClassifierLoss = criterion(
                    predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
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
            # if((i+1) % 10 == 0 ):
            #   classifier.eval()
            #   # accuracy
            #   _, predicted = torch.max(predictions, 1)
            #   total_train += b_labels.size(0)
            #   correct_train += predicted.eq(b_labels.data).sum().item()
            #   train_accuracy = 100 * correct_train / total_train
            #   print("({}) train_accuracy : {}".format(i+1,train_accuracy))
            #   classifier.train()

        log_print("Epoch " + str(epoch_i+1) + " Complete")
        log_print("Epoch Time : ", time.time()-t0)
        evaluation(epoch_i)
        validation_acc = validate(epoch_i)
        save_best_model(models_path, epoch_i, validation_acc)
        save_params(epoch_i, models_path)


def per_label_accuracy(b_labels, predicted, class_accuracies):
    labels = torch.unique(b_labels)
    for label in labels:
        label_indexes = torch.where((b_labels == label))[0]
        number_of_label = len(label_indexes)
        predicts = [predicted[i].item() for i in label_indexes]
        true_predicts = torch.sum(
            (torch.tensor(predicts).to(device) == label)).item()
        # print(f"predicts : {predicts}")
        # print(f"label : {label}")
        # print(f"true_predicts : {true_predicts}")
        lb = label.item()
        if lb in class_accuracies.keys():
            class_accuracies[lb]["true_predict"] = true_predicts + \
                class_accuracies[lb]["true_predict"]
            class_accuracies[lb]["total"] = number_of_label + \
                class_accuracies[lb]["total"]
        else:
            class_accuracies[lb] = {
                "true_predict": true_predicts, "total": number_of_label}


def print_validation_per_class_accuracy_validation(class_accuracies):
    total_label_base_accuracy_validation.append(class_accuracies)
    for x in class_accuracies:
        acc = (class_accuracies[x]["true_predict"] /
               class_accuracies[x]["total"])*100
        # print(
        #     f"{class_accuracies[x]['true_predict']} / {class_accuracies[x]['total']} >> accuracy class {x} : {acc}")


def print_validation_per_class_accuracy_evaluation(class_accuracies):
    total_label_base_accuracy_evaluation.append(class_accuracies)
    for x in class_accuracies:
        acc = (class_accuracies[x]["true_predict"] /
               class_accuracies[x]["total"])*100
        # print(
        #     f"{class_accuracies[x]['true_predict']} / {class_accuracies[x]['total']} >> accuracy class {x} : {acc}")


def print_validation_per_class_accuracy_test(class_accuracies):
    total_label_base_accuracy_test.append(class_accuracies)
    for x in class_accuracies:
        acc = (class_accuracies[x]["true_predict"] /
               class_accuracies[x]["total"])*100
        # print(
        #     f"{class_accuracies[x]['true_predict']} / {class_accuracies[x]['total']} >> accuracy class {x} : {acc}")


def validate(epoch):
    # print("call validate")
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        class_accuracies = {}
        for data in validation_dataloader:

            b_input_ids = data[0].to(device)
            b_input_mask = data[1].to(device)
            b_labels = data[2].to(device)

            model_outputs = transformer(
                b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            outputs = classifier(hidden_states)
            _, predicted = torch.max(outputs.data, 1)

            total += b_labels.size(0)
            correct += (predicted == b_labels).sum().item()

            per_label_accuracy(b_labels, predicted, class_accuracies)

    accuracy = (correct / total) * 100
    log_print(f"validation Accuracy : {accuracy}")
    print_validation_per_class_accuracy_validation(class_accuracies)
    print_validation_accuracy(epoch+1, accuracy)
    # print("validate : {} / {} * 100 = {} ".format(correct, total, accuracy))
    classifier.train()
    return accuracy


def evaluation(epoch):
    # print("call evaluation")

    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        class_accuracies = {}
        for data in train_dataloader:

            b_input_ids = data[0].to(device)
            b_input_mask = data[1].to(device)
            b_labels = data[2].to(device)
            model_outputs = transformer(
                b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            outputs = classifier(hidden_states)
            _, predicted = torch.max(outputs.data, 1)
            # print(f"predicted : {predicted}")
            # print(f"b_labels : {b_labels}")
            total += b_labels.size(0)
            correct += (predicted == b_labels).sum().item()
            per_label_accuracy(b_labels, predicted, class_accuracies)

    accuracy = (correct / total) * 100
    log_print(f"evaluation Accuracy : {accuracy}")

    # print("class_accuracies : ", class_accuracies)
    print_evaluation_accuracy(epoch+1, accuracy)
    print_validation_per_class_accuracy_evaluation(class_accuracies)
    # print("evaluation : {} / {} * 100 = {} ".format(correct, total, accuracy))
    classifier.train()
    return accuracy


def test(transformer, classifier):
    log_print()
    log_print("Start Testing ...")
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        class_accuracies = {}
        for data in test_dataloader:

            b_input_ids = data[0].to(device)
            b_input_mask = data[1].to(device)
            b_labels = data[2].to(device)
            model_outputs = transformer(
                b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            outputs = classifier(hidden_states)
            _, predicted = torch.max(outputs.data, 1)
            total += b_labels.size(0)
            correct += (predicted == b_labels).sum().item()
            per_label_accuracy(b_labels, predicted, class_accuracies)

    accuracy = (correct / total) * 100
    log_print(f"Test Accuracy : {accuracy}")

    print_test_accuracy(accuracy)
    print_validation_per_class_accuracy_test(class_accuracies)
    # print("test : {} / {} * 100 = {} ".format(correct, total, accuracy))
    return accuracy


def print_results(train_acc, validation_acc, test_acc):
    train_acc.insert(0, "train")
    validation_acc.insert(0, "validation")
    test_acc.insert(0, "test")

    execl_path = default_path_str + dir_name + f"/{dir_name}.xlsx"
    workbook = xlsxwriter.Workbook(execl_path)
    worksheet = workbook.add_worksheet()

    epochs = [x for x in range(len(train_acc))]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df = pd.DataFrame(
            data=[train_acc, validation_acc, test_acc], columns=epochs)
        width = 640
        height = 360
        for i in range(len(df)):
            for j in range(len(df.iloc[i])):
                if not (pd.isnull(df.iloc[i, j])):
                    worksheet.write(i, j, df.iloc[i, j])
            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'name': ['sheet1', i, 0],
                'values':     ['sheet1', i, 1, i, len(df.iloc[i])-1],
            })
            # chart.set_title({
            #     'name': [i,0],
            # })
            chart.set_size({'width': width, 'height': height})
            worksheet.insert_chart(
                3, 0, chart, {'x_offset': i*(width), 'y_offset': (height)})
        workbook.close()

        # df.to_excel(execl_path, index=False)
        # print(df)


def change_format(lst):
    rep = {}
    for i, x in enumerate(lst):
        for item in x.keys():
            if item in rep.keys():
                predict = x[item]["true_predict"]
                total = x[item]["total"]
                acc = (predict/total)*100
                rep[item].append([predict, total, acc])
            else:
                predict = x[item]["true_predict"]
                total = x[item]["total"]
                acc = (predict/total)*100
                rep[item] = [[predict, total, acc]]
    return rep


def add_chart(workbook, worksheet, s_row, s_col, e_row, e_col, name, lbl, x, y, row, col):
    width = 640
    height = 360
    chart = workbook.add_chart({'type': 'line'})
    chart.add_series({
        'name': ['sheet1', s_row-2, s_col-1],
        'values':     ['sheet1', s_row, s_col, e_row, e_col],
    })
    chart.set_title({
        'name': f"{name}_{lbl}",
    })
    chart.set_size({'width': width, 'height': height})
    worksheet.insert_chart(
        row, col, chart, {'x_offset': x*(width), 'y_offset': y*(height)})


def print_per_class(train_per_lbl_acc, validation_per_lbl_acc, test_per_lbl_acc):
    len_epoch = len(train_per_lbl_acc)
    reformat_epla = change_format(train_per_lbl_acc)
    reformat_vpla = change_format(validation_per_lbl_acc)
    reformat_tpla = change_format(test_per_lbl_acc)

    execl_path = default_path_str + dir_name + f"/{dir_name}_per_label.xlsx"
    workbook = xlsxwriter.Workbook(execl_path)
    worksheet = workbook.add_worksheet()

    x_row = 1
    y_col = len(reformat_epla[list(reformat_epla.keys())[0]])
    row = 1
    for i, x in enumerate(sorted(reformat_epla)):
        col = 1
        add_chart(workbook, worksheet, row+2, col+1,
                  row+2, col+1+(len_epoch), "Train", x, 1, i, x_row, y_col)
        for i, y in enumerate(reformat_epla[x]):
            worksheet.write(0, col+1, i+1)
            predict = y[0]
            total = y[1]
            acc = y[2]
            worksheet.write(row, col + 1, predict)
            worksheet.write(row + 1, col + 1, total)
            worksheet.write(row + 2, col + 1, acc)
            worksheet.merge_range(f"B{row+1}:B{row+3}", f"{x}")
            col += 1
        row += 3
    row = row+1
    for i, x in enumerate(sorted(reformat_vpla)):
        col = 1
        add_chart(workbook, worksheet, row+2, col+1,
                  row+2, col+1+(len_epoch), "validation", x, 2, i, x_row, y_col)
        for i, y in enumerate(reformat_vpla[x]):
            worksheet.write(0, col+1, i+1)
            predict = y[0]
            total = y[1]
            acc = y[2]
            worksheet.write(row, col + 1, predict)
            worksheet.write(row + 1, col + 1, total)
            worksheet.write(row + 2, col + 1, acc)
            worksheet.merge_range(f"B{row+1}:B{row+3}", f"{x}")
            col += 1
        row += 3
    row = row+1
    for i, x in enumerate(sorted(reformat_tpla)):
        col = 1
        for i, y in enumerate(reformat_tpla[x]):
            worksheet.write(0, col+1, i+1)
            predict = y[0]
            total = y[1]
            acc = y[2]
            worksheet.write(row, col + 1, predict)
            worksheet.write(row + 1, col + 1, total)
            worksheet.write(row + 2, col + 1, acc)
            worksheet.merge_range(f"B{row+1}:B{row+3}", f"{x}")
            col += 1
        row += 3
    workbook.close()


train(train_dataloader)


transformer, classifier = load_best_model(models_path)
classifier.cuda()
transformer.cuda()
if transformer == False:
    log_print("an error occurred : load best model failed")
    exit()
test(transformer, classifier)

print_results(total_acc_evaluation, total_acc_validation, total_acc_test)
print_per_class(total_label_base_accuracy_evaluation,
                total_label_base_accuracy_validation, total_label_base_accuracy_test)
