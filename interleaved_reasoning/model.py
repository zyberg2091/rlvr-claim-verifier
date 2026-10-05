"""Model initialization and the original PPO value-head wrapper."""

import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class PPO_Model(nn.Module):

  def __init__(self, lm):
    super().__init__()
    self.lm = lm
    self.value_head = nn.Linear(lm.config.hidden_size, 1)

  def forward(self, input_ids, past_key_values, use_cache):
    # Pass past_key_values and set use_cache=True for efficient autoregressive decoding
    outputs = self.lm(input_ids, output_hidden_states=True, past_key_values=past_key_values, use_cache=use_cache)

    logits = outputs.logits
    # Ensure last_hidden is on the same device as value_head AND cast to float32
    # outputs.hidden_states[-1] will be (batch_size, sequence_length, hidden_size)
    # If only the last token is passed as input_ids (with past_key_values), sequence_length will be 1.

    if use_cache == True:
      last_hidden = outputs.hidden_states[-1][:, -1, :]
      last_hidden = last_hidden.to(self.value_head.weight.device).float()
      value = self.value_head(last_hidden).squeeze(-1)

      return logits, value, outputs.past_key_values # Return updated past_key_values

    else:
      last_hidden = outputs.hidden_states[-1]
      last_hidden = last_hidden.to(self.value_head.weight.device).float()
      value = self.value_head(last_hidden).squeeze(-1)

      return logits, value


def load_policy():
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        pass

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    # Define BitsAndBytesConfig for 8-bit quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load in 8-bit with torch_dtype=torch.float16 to avoid casting warnings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    # Explicitly set output_hidden_states to True in the model's config
    model.config.output_hidden_states = True

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Add a new pad token and resize model embeddings correctly
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Ensure the model config reflects the new pad token id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Lora technique
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(r=32,  # LoRA rank
                            lora_alpha=16,  # Scaling factor
                            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],  # Target layers for Qwen2.5
                            bias="none",
                            task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)
    policy_model = PPO_Model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_model.to(device)

    return model, tokenizer, policy_model, device
