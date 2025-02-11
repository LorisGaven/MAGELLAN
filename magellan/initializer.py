'''

'''

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lamorel import BaseModelInitializer
from typing import List


class SequentialInitializer(BaseModelInitializer):
    
    def __init__(self, initializers:List[BaseModelInitializer]):
        super().__init__()
        self._initializers = initializers
        
    def initialize_model(self, model):
        for _initializer in self._initializers:
            model = _initializer.initialize_model(model)
        return model

class WeightsLoaderInitializer(BaseModelInitializer):
    
    def __init__(self, weights_path):
        super().__init__()
        self._weights_path = weights_path
        
    def initialize_model(self, model):
        if self._weights_path is not None:
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint")
            hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
            model.load_state_dict(state_dict=hf_llm_module_dict, strict=False)
            print(f"Model loaded from {self._weights_path}")
        return model

class PeftInitializer(BaseModelInitializer):
    
    def __init__(self, model_type, model_name, use_lora, use_4bit, r, alpha, use_cache=True):
        super().__init__()
        self._model_type = model_type
        self._model_name = model_name
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._use_cache = use_cache

    def _get_model_config(self):
        if "t5" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q", "v"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        elif "opt" in self._model_name or "Llama" in self._model_name or "Mistral" in self._model_name or "Qwen" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            raise NotImplementedError()

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']
            if self._model_type == "seq2seq" or not self._use_cache:
                llm_module.gradient_checkpointing_enable()  # reduce number of stored activations
            if self._use_4bit:
                llm_module = prepare_model_for_kbit_training(llm_module)

            # Init adapters
            config = self._get_model_config()
            peft_model = get_peft_model(llm_module, config)
            
            # Add delayed adapers
            peft_model.add_adapter("delayed_adapters", config)
            
            # Add sr adapters
            peft_model.add_adapter("sr_adapters", config)
            
            # Add critic target adapters
            peft_model.add_adapter("critic_target", config)
            
            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None
                        
                if 'lm_head' in name:
                    param.requires_grad_(False)
                    
                if name.split('.')[-1] == "delayed_adapters" and hasattr(param, "weight"):  # Freeze delayed parameters
                    param.weight.requires_grad = False
                    
                if name.split('.')[-1] == "critic_target" and hasattr(param, "weight"):  # Freeze critic target parameters
                    param.weight.requires_grad = False
                    
                if (name.split('.')[-1] == "default" or name.split('.')[-1] == "sr_adapters") and hasattr(param, "weight"):
                    param.weight.requires_grad = True
                
            model._modules['_LLM_model'] = peft_model
        model.eval()
        model._modules['_LLM_model'].config.use_cache = self._use_cache
        return model