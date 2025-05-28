
import torch 
import torch.nn as nn
from transformers import GPT2LHeadModel, GPT2 tokenizer, Trainer, TrainingArguments
from data sets import load_dataset
import pickle
import os
from typing import List, Dict

class SimpleAIChatbot:
  """ 
  a simplified AI chabot simalar to Deepseek using Gpt -2 architecture.
  """

def __init__(self, model_name: str = "gpt2" , device : str = None):
  """
  initilize chatbot with a pre=trained model.

  Args:
  model_name: Name of the pre_trained model.
