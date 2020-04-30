#!/usr/bin/env python
# -*- coding: utf-8 -*-
import load
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Loads list of questions from dataset
questions = load.data()

# Tokenize input
tokenized_questions = [tokenizer.tokenize(q) for q in questions]

# Convert token to vocabulary indices
indexed_tokens = [tokenizer.convert_tokens_to_ids(tq) for tq in tokenized_questions]

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = []
for tokenized_question in tokenized_questions:
    length = len(tokenized_question)
    first_sep = tokenized_question.index('[SEP]')+1
    segments_ids.append([0]*(first_sep) + [1]*(length-first_sep))

# Convert inputs to PyTorch tensors
tokens_tensors = [torch.tensor([tokens]) for tokens in indexed_tokens]
segments_tensors = [torch.tensor([segments]) for segments in segments_ids]

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

### @TODO left off here ###

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'