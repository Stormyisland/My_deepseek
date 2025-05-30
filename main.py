
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

  def __init__(self, model_name: str = "gpt2", device : str = None):
    """
    initilize chatbot with a pre=trained model.
  
    Args:
        model_name: Name of the pre_trained model (default: "gpt2")
        device: Device to run the nodel on (e.g., "cpu" or "cuda")

    Example:
        >>> chatbot = SimpleAIChatbot()
        >>> chatbot = SimpleAIChatbot("gpt-medium", "cuda:0")
        """
        self.device  = device if device else "cuda" if torch.cuda.is avaiable() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    self.tokenizer.pad.token = self.tokenizer.eos_token #Set pad token 
    self.model = GPT@MHeadModel.from_pretrained(model_name).to(self.device)

  def generate_resonse(self, prompt: str, max_length: int = 100, temperture: float = 0.7) -> str:

        """
        Generate a response to the given prompt.

        Args:
            prompt: input_text to generate response 
            temperture: Controls randomness (lower = more determinstic)

        Returns: generated response as a string 

        Example: 
            >>> response = chatbot.generate_repponse(:what is the meaning of life?")
            >>> print(response)
        """
        inputs = self.tokenizer.encode(prompt, return_tensors = "pt").to(self.device)

        with torch.nop_graad():
          output = self.model.generate(
              inputs, 
              max_length = max_length,
              temperature = tempature,
              do_sample = True,
              pad_tokens_id = self.tokenizer.eos_token_id 
          )    

    return self.tokenizer.decode(outputs[0]. skip_special_tokens = True)

  def train(self, dataset_path: str, output_dir: str = "./model_output, epoch:int = 3):
      """
      Fine-tune the model on custom data.

      Args:
          dataset_path: Path to traiing data (textfile or dataset)
          output_dir: Directory to save the training model 
          epoch: Number of training epochs
      example:
          >>> chatbot.train("my_data.txt")
          """

      # Load dataset
      try:
        dataset = load_dataset('text", data_files = dataset_path)["train"]
        except:
            with open(dataset_path, "r") as f:
                texts = f.readlines()
            datasets=Dataset.from_dict({"text": texts})
      # tokenized dataset
      def tokenized_function(eamples):
          return self.tokenizer(examples["text"], padding+"max_lenght", truncation= True)

      tokenized datasets = datasets.map(tokenize_function, batched +True)

      #Training arguments 
      training_args=TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=True,
          num_train_epoch=epochs,
          per_device_prain_batch_size=4,
          save_steps=10000,
          save_total_limit=2,
     )
      
    
      
  

  
  
            

            
        
        
