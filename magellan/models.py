'''
    This file contains the implementation of the SAC's models 
    "LogScoringModuleFn" is the actor and "ValueHeadModuleFn" is the critic.
'''

import torch
from lamorel import BaseModuleFunction


# LLM Actor
class LogScoringModuleFn(BaseModuleFunction):
    
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position + 1:]
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        tokens_logprobs = \
            torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()
    
    def get_parameters_name_filter(self):
        return lambda n: '.default.' in n #or '.lm_head' in n # LoRA default params + lm head


# Crtic on the last hidden state of the LLM decoder
class ValueHeadModuleFn(BaseModuleFunction):
    
    def __init__(self, model_type, pre_encoded_input, name):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input
        self._name = name
        self._target= "target" in name
    
    def initialize(self):
        if 'hidden_size' in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")
        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        ).to(self.device)
        
        self.value_head_op.requires_grad_(not self._target)
    
    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size
            model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]
            
        emb = model_head.to(torch.float32).to(self.device)
        if 'emb' in kwargs and kwargs['emb']:
            return emb.cpu()
        
        value = self.value_head_op(emb).cpu()
        
        return value
    
    def get_parameters_name_filter(self):
        # LoRA default params + MLP head
        if self._target:
            return lambda n: f'.{self._name}.' in n or '.target.' in n
        else:
            return lambda n: f'.{self._name}.' in n or '.default.' in n

# SR estimation head for MAGELLAN       
class SRHeadModuleFn(BaseModuleFunction):
    
    def __init__(self, model_type, pre_encoded_input, name, adapters, train_llm):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input
        self._name = name
        self._delayed = "delayed" in name
        self._adapters = adapters
        self._train_llm = train_llm
    
    def initialize(self):
        if 'hidden_size' in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")
        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        ).to(self.device)
        
        self.value_head_op.requires_grad_(not self._delayed)
    
    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size
            model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]
        
        emb = model_head.to(torch.float32).to(self.device)
        
        if 'emb' in kwargs and kwargs['emb']:
            return emb.cpu()
        
        sr = self.value_head_op(emb)
        return sr.cpu()
    
    def get_parameters_name_filter(self):
        # LoRA default params + MLP head
        if self._delayed:
            return lambda n: f'.{self._name}.' in n or '.delayed_adapters.' in n
        else:
            return lambda n: f'.{self._name}.' in n or f'.{self._adapters}.' in n