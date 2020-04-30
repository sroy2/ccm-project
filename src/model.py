#!/usr/bin/env python
# -*- coding: utf-8 -*-
import load
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging

class MaskedTokenBert:
    '''Class for predicting masked tokens
    '''
    def __init__(self, bert='bert-base-uncased', file='masked_data.txt', debug=False):
        logging.basicConfig(level=logging.ERROR)
        if debug:
            logging.basicConfig(level=logging.INFO)
        # Load pre-trained model tokenizer (vocabulary)
        self.bert = bert
        self.file = file
        self.tokenizer = BertTokenizer.from_pretrained(self.bert)

    def _load(self, file=None):
        # Loads list of questions from dataset
        self.file = file if file else self.file
        self.data = load.data(file)
        self.size = len(self.data)

    def _load_line(self, line):
        self.data = [line]
        self.size = 1
        
    def tokenize_data(self, file=None):
        '''Tokenizes self.data
        '''
        # Check that data is loaded
        try:
            if self.data:
                pass
        except:
            self._load(file) if file else self._load(self.file)
            
        # Tokenize input
        self.t_data = [self.tokenizer.tokenize(i) for i in self.data]
    
        # Convert token to vocabulary indices
        self.t_idx = [self.tokenizer.convert_tokens_to_ids(i) for i in self.t_data]
    
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        self.segment_ids = []
        self.masks = []
        for i in self.t_data:
            length = len(i)
            first_sep = i.index('[SEP]')+1
            self.segment_ids.append([0]*(first_sep) + [1]*(length-first_sep))
            self.masks.append([x for x, m in enumerate(i) if m == '[MASK]'])
        
        # Convert inputs to PyTorch tensors
        if torch.cuda.is_available():
            # If you have a GPU, put everything on cuda
            self.token_tensors = [torch.tensor([i]).to('cuda') for i in self.t_idx]
            self.segment_tensors = [torch.tensor([i]).to('cuda') for i in self.segment_ids]
        else:
            self.token_tensors = [torch.tensor([i]) for i in self.t_idx]
            self.segment_tensors = [torch.tensor([i]) for i in self.segment_ids]

    def model(self):
        '''Applies pre-trained model to self.data
        '''
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(self.bert)
        self.mask_model = BertForMaskedLM.from_pretrained(self.bert)

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.model.eval()
        self.mask_model.eval()
        
        # If you have a GPU, put everything on cuda
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.mask_model.to('cuda')
        
        # Check that data has been tokenized
        try:
            if self.t_data:
                pass
        except:
            self.tokenize_data()
        
        # Predict hidden states features for each layer
        self.outputs = []
        self.encoded_layers = []
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            for i in range(self.size):
                self.outputs.append(self.model(input_ids=self.token_tensors[i],
                                               token_type_ids=self.segment_tensors[i]))
                # First element is the hidden state of the last Bert model layer
                self.encoded_layers.append(self.outputs[i][0])
        
        # Predict best word for each masked token
        self.predictions = []
        with torch.no_grad():
            for i in range(self.size):
                # First element is all masked predictions
                self.predictions.append(self.mask_model(input_ids=self.token_tensors[i],
                                                        token_type_ids=self.segment_tensors[i])[0])

        # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
        assert tuple(self.encoded_layers[0].shape) == (1, len(self.t_idx[0]), self.model.config.hidden_size)

    def _decode(self, i):
        return self.tokenizer.convert_ids_to_tokens([i])[0]
    
    def predict(self, top_n=5):
        '''Predicts the top_n (default=5) candidates for each [MASK]
        '''
        # Check we have a model to predict from
        try:
            if self.predictions:
                pass
        except:
            self.model()
        
        self.p_idx = []
        self.p_items = []
        for i in range(self.size):
            idx = []
            items = []
            for m in self.masks[i]:
                pred_idx = torch.argsort(-self.predictions[i][0, m])[:top_n]
                idx.append([x.item() for x in pred_idx])
                items.append([self._decode(x.item()) for x in pred_idx])
            self.p_idx.append(idx)
            self.p_items.append(items)

        return self.p_items
