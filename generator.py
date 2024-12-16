import warnings
from typing import List, Dict, Union, Optional, Any, Tuple

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LlamaForCausalLM, 
    Qwen2ForCausalLM,
    MistralForCausalLM,
    Gemma2ForCausalLM, 
    T5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

HF_TOKEN = "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"
SUPPORTED_DECODER_ONLY_GENERATORS = [LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Gemma2ForCausalLM]
SUPPORTED_ENCODER_DECODER_GENERATORS = [T5ForConditionalGeneration]


def load_model_in_4bit(cls, model_name_or_path):

    from transformers import BitsAndBytesConfig
    model = cls.from_pretrained(
        model_name_or_path, 
        cache_dir=None, 
        device_map = "auto", 
        max_memory = {0: "800000MB"}, 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
        token=HF_TOKEN, 
        torch_dtype=torch.bfloat16, 
    )
    return model

def load_llm_tokenizer_and_model(model_name, padding_side="left", dtype=torch.bfloat16, load_in_4bit=False, device=None):

    device = device or torch.device("cuda")
    MODEL_MAP = {
        # llama
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B", 
        "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3_70b_instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3.1_70b_instruct": "meta-llama/Llama-3.1-70B-Instruct",
        # Mistral 
        "mistral_7b": "mistralai/Mistral-7B-v0.1", 
        "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.2", 
        # Qwen 
        "qwen2": "Qwen/Qwen2-7B-Instruct",
        "qwen2.5_7b": "Qwen/Qwen2.5-7B",
        "qwen2.5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5_32b_instruct": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5_72b_instruct": "Qwen/Qwen2.5-72B-Instruct",
        # Gemma
        "gemma2_2b": "google/gemma-2-2b", 
        "gemma2_2b_itstruct": "google/gemma-2-2b-it",
        "gemma2_9b": "google/gemma-2-9b", 
        "gemma2_9b_instruct": "google/gemma-2-9b-it",
        "gemma2_27b_instruct": "google/gemma-2-27b-it",
    }
    
    if model_name not in MODEL_MAP:
        raise ValueError(f"{model_name} is not a valid model name. Current available models: {list(MODEL_MAP.keys())}")
    
    model_name_or_path = MODEL_MAP[model_name]
    
    padding_side = "left"
    print(f"loading tokenizer for \"{model_name_or_path}\" with padding_side: \"{padding_side}\"")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, token=HF_TOKEN)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Missing padding token, setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if load_in_4bit:
        print(f"loading \"{model_name_or_path}\" model in 4-bits ...")
        # 在用4-bit加载模型的时候不需要用model.to(device), 因为在加载的时候已经指定了
        model = load_model_in_4bit(AutoModelForCausalLM, model_name_or_path)
    else:
        print(f"loading \"{model_name_or_path}\" model in {dtype} ...")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, token=HF_TOKEN)
        # model.to(device)
        model.to('mps')
    model.eval()

    return tokenizer, model 

def load_t5_tokenizer_and_model(model_name, dtype=torch.bfloat16, load_in_4bit=False, device=None):

    device = device or torch.device("cuda")
    MODEL_MAP = {
        "flan_t5_large": "google/flan-t5-large", 
        "flan_t5_xl": "google/flan-t5-xl", 
        "flan_t5_xxl": "google/flan-t5-xl",
    }
    if model_name not in MODEL_MAP:
        raise ValueError(f"{model_name} is not a valid model name. Current available models: {list(MODEL_MAP.keys())}")
    
    model_name_or_path = MODEL_MAP[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if load_in_4bit:
        print(f"loading \"{model_name_or_path}\" model in 4-bits ...")
        model = load_model_in_4bit(T5ForConditionalGeneration, model_name_or_path)
    else:
        print(f"loading \"{model_name_or_path}\" model in {dtype} ...")
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        model.to(device)
    model.eval()

    return tokenizer, model

def to_device(inputs, device):

    def dict_to_device(data):
        return {k: item.to(device) if torch.is_tensor(item) else item for k, item in data.items()}
    
    if isinstance(inputs, (tuple, list)):
        new_data = [] 
        for item in inputs:
            if isinstance(item, dict):
                new_data.append(dict_to_device(item))
            elif torch.is_tensor(item):
                new_data.append(item.to(device))
            else:
                new_data.append(item)
    elif isinstance(inputs, dict):
        new_data =dict_to_device(inputs)
    else:
        raise TypeError(f"Currently do not support using <{type(inputs)}> as the type of a batch")

    return new_data

def pad_token_ids(token_ids: Tensor, max_length: int, pad_token_id: int) -> Tensor:

    batch_size, num_tokens = token_ids.shape 
    if num_tokens >= max_length:
        return token_ids[:, :max_length]
    padding_length = max_length - num_tokens
    dtype = token_ids.dtype
    device = token_ids.device 
    padding_tensor = torch.zeros((batch_size, padding_length), dtype=dtype).fill_(pad_token_id).to(device)
    padded_token_ids = torch.cat([token_ids, padding_tensor], dim=1)

    return padded_token_ids

def pad_token_logits(token_logits: Tensor, max_length: int) -> Tensor:

    batch_size, num_tokens, vocab_size = token_logits.shape 
    if num_tokens >= max_length:
        return token_logits[:, :max_length]
    padding_length = max_length-num_tokens
    dtype, device = token_logits.dtype, token_logits.device
    padding_tensor = torch.zeros((batch_size, padding_length, vocab_size), dtype=dtype).to(device)
    padded_token_logits = torch.cat([token_logits, padding_tensor], dim=1)
    
    return padded_token_logits

class Generator(nn.Module):

    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        generator: AutoModelForCausalLM, 
        max_length: int=4096, 
        max_new_tokens: int=128,
        batch_size: int=4, 
        **kwargs
    ):
        super().__init__()

        supported_generator_types = tuple(SUPPORTED_DECODER_ONLY_GENERATORS+SUPPORTED_ENCODER_DECODER_GENERATORS)
        assert isinstance(generator, supported_generator_types) # currently only support using LLaMA3 or Qwen2 as the generator

        self.tokenizer = tokenizer
        self.generator = generator
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.check_tokenizer_padding()

        self.is_chat = kwargs.get("is_chat", None) or self.init_is_chat()
        self.is_encoder_decoder = kwargs.get("is_encoder_decoder", None) or self.init_is_encoder_decoder()
        self.config = self.generator.config 
        self.config.update(kwargs)
    
    @property
    def device(self):
        return self.generator.device 
    
    @property
    def dtype(self):
        return self.generator.dtype 

    def init_is_chat(self):
        model_name_or_path = self.generator.config._name_or_path.lower()
        if "instruct" in model_name_or_path or "chat" in model_name_or_path or "-it" in model_name_or_path:
            is_chat = True
        else:
            is_chat = False
        return is_chat
    
    def init_is_encoder_decoder(self):
        if isinstance(self.generator, tuple(SUPPORTED_ENCODER_DECODER_GENERATORS)):
            is_encoder_decoder = True 
        elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            is_encoder_decoder = False
        else:
            raise ValueError(f"{type(self.generator)} is an unknow generator!")
        return is_encoder_decoder
    
    def check_tokenizer_padding(self):
        if isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
                raise ValueError("pad_token or pad_token_id is None in the tokenizer of generator. suggest to set pad_token and pad_token_id to eos_token and eos_token_id respectively.")
            if self.tokenizer.padding_side == "right":
                raise ValueError("Dected right padding using decoder-only transformers as the generator, which may cause some errors. It is suggested to use \"left\" padding!")
    
    def get_generator_prompts_chat_format(
        self, 
        instructions: List[str], 
        messages: Union[List[List[dict]], List[str]],
        **kwargs
    ) -> List[List[Dict[str, str]]]:
        """
        Input: 
            instruction: [str]
            messages: [str] or [[{"user": "user_content"}, {"assistant": "assistant_content"}],...]
        Output:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        """
        prompts = [] 
        assert len(instructions) == len(messages) # number of instructions shoule be the same as messages 
        for instruction, message_list in zip(instructions, messages):
            if isinstance(self.generator, (LlamaForCausalLM, Qwen2ForCausalLM)):
                one_prompt = [{"role": "system", "content": instruction}]
                if isinstance(message_list, str):
                    one_prompt.append({"role": "user", "content": message_list})
                elif isinstance(message_list, list):
                    assert "user" in message_list[0] # # the first message must comes from user in the form of: {"user": "user_message"}
                    for message in message_list:
                        if "user" in message:
                            one_prompt.append({"role": "user", "content": message["user"]})
                        # if "system" in message:
                        #     one_prompt.append({"role": "system", "content": message["system"]})
                        if "assistant" in message:
                            one_prompt.append({"role": "assistant", "content": message["assistant"]})
                else:
                    raise ValueError(f"Invalid message type: {type(message_list)}. Only support str or List[dict] messages")
                prompts.append(one_prompt)
            elif isinstance(self.generator, (MistralForCausalLM, Gemma2ForCausalLM)):
                # Mistral Don't have System Role 
                if isinstance(message_list, str):
                    one_prompt = [{"role": "user", "content": instruction + "\n\n" + message_list}]
                elif isinstance(message_list, list):
                    assert "user" in message_list[0] # the first message must comes from user in the form of: {"user": "user_message"}
                    one_prompt = [{"role": "user", "content": instruction + "\n\n" + message_list[0]["user"]}]
                    for message in message_list[1:]:
                        if "user" in message:
                            one_prompt.append({"role": "user", "content": message["user"]})
                        if "assistant" in message:
                            one_prompt.append({"role": "assistant", "content": message["assistant"]})
                else:
                    raise ValueError(f"Invalid message type: {type(message_list)}. Only support str or List[dict] messages")
                prompts.append(one_prompt)
            else:
                raise NotImplemented(f"chat format for {type(self.generator)} is not implemented yet!")
        return prompts
    
    def tokenizer_encode_chat_format(self, prompts: List[List[Dict[str, str]]], max_length: int=None, add_generation_prompt: bool=True, **kwargs) -> Dict[str, Tensor]:
        """
        Params:
            add_generation_prompt: 为True的时候会在token的最后添加<|start_header_id|>assistant<|end_header_id|>, 为False的时候以<|eot_id|>结尾
        Input:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        Output:
            {"input_ids": Tensor, "attention_mask": Tensor}
        """
        max_length = self.max_length if max_length is None else max_length
        # apply_chat_template 会根据是user还是assistant的输入来添加特殊的token, 比如在LLaMA3的格式是:
        # <begin_of_text><|start_header_id|>system<|end_header_id|>\n\n 指令输入 <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n 用户输入 <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=add_generation_prompt) 
        # 下面这里默认添加了特殊的tokens, 比如LLaMA3会在开头添加<begin_of_text> token, 而T5会在最后添加</s>
        batch_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def tokenizer_encode(self, prompts: List[str], max_length: int=None, **kwargs) -> Dict[str, Tensor]:
        max_length = self.max_length if max_length is None else max_length
        # 下面这里默认添加了特殊的tokens, 比如LLaMA3会在开头添加<begin_of_text> token, 而T5会在最后添加</s>
        batch_dict = self.tokenizer(prompts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def get_generated_token_ids(self, input_ids: Tensor, token_ids: Tensor) -> Tensor:
        if isinstance(self.generator, T5ForConditionalGeneration): # 不清楚BART是否也是第一个是其他的token，所以只写T5
            generated_token_ids = token_ids[:, 1:] # T5模型第一个token是<bos> token
        elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            generated_token_ids = token_ids[:, input_ids.shape[1]:]
        else:
            raise NotImplementedError(f"get_generated_token_ids is not implemented for {type(self.generator)}!")
        return generated_token_ids
    
    def greedy_generate(
        self, 
        inputs: Dict[str, Tensor],
        pad_to_max_new_tokens: bool=False,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Inputs: 
        {"input_ids": Tensor, "attention_mask": Tensor}

        Outputs:
        Tensor, Tensor
        """
        # batch_size, max_new_tokens, device = self.batch_size, self.max_new_tokens, self.device
        device = self.device
        batch_size = kwargs.get("batch_size", None) or self.batch_size
        max_new_tokens = kwargs.get("max_new_tokens", None) or self.max_new_tokens
        stopping_criteria = kwargs.get("stopping_criteria", None)

        generated_token_ids_list, generated_token_logits_list = [], [] 
        for i in range((len(inputs["input_ids"])-1)//batch_size+1):
            batch_inputs = {k: v[i*batch_size: (i+1)*batch_size] for k, v in inputs.items()}
            batch_inputs = to_device(batch_inputs, device)
            batch_outputs = self.generator.generate(
                **batch_inputs, 
                max_new_tokens=max_new_tokens, 
                output_scores=True, 
                return_dict_in_generate=True, 
                do_sample=False, 
                temperature=1.0,
                stopping_criteria=stopping_criteria,
            ) # temperature=1.5, do_sample=True)
            # batch_generated_token_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()
            batch_generated_token_ids = self.get_generated_token_ids(batch_inputs["input_ids"], batch_outputs.sequences).detach().cpu()
            batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores], dim=1).detach().cpu()
            
            generated_token_ids_list.append(batch_generated_token_ids)
            generated_token_logits_list.append(batch_generated_token_logits)
        
        max_generation_length = max_new_tokens if pad_to_max_new_tokens else \
            max([x.shape[-1] for x in generated_token_ids_list])
        generated_token_ids_list = [
            pad_token_ids(
                token_ids, 
                max_length=max_generation_length, 
                pad_token_id=self.tokenizer.pad_token_id
            ) 
            for token_ids in generated_token_ids_list
        ]
        generated_token_logits_list = [
            pad_token_logits(
                token_logits, 
                max_length=max_generation_length
            ) 
            for token_logits in generated_token_logits_list
        ]

        generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
        generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

        return generated_token_ids, generated_token_logits
        
    def generate(self, inputs, **kwargs) -> Tuple[Tensor, Tensor]:

        """
        目前支持的kwargs中的变量有:
        max_tokens/max_new_tokens: int, 
        batch_size: int, 
        stop_words: str / [str] 当模型在生成的过程中生成stop_words中的token的时候就停止, 只传一个str的时候会把每一个character当做stop words
        """

        max_new_tokens = kwargs.get("max_tokens", None) or kwargs.get("max_new_tokens", None)
        if max_new_tokens is None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        batch_size = kwargs.get("batch_size", None)
        if batch_size is None:
            kwargs["batch_size"] = self.batch_size
        
        return self.greedy_generate(inputs, **kwargs)
