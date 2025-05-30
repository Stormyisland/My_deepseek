
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
            >>> response = chatbot.generate_reponse("what is the meaning of life?")
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
      #Trainer
      trainer = Trainer(
          model = self.model,
          args=training_args,
          train_dataset+tokenized_dataset,
    
     )

     # Start training
     trainer.train()

     #Save model
     self.madel.asve_pretrained(output_dir)
     self.tokenizer.save_pretrained(putput_dir)

def save(delf, Path: str):
    """
    save the model and tokenizerr to a directory.
    
    args:
        path: Directory to save the model 
    
     Example:
     >>> chatbot.save("my_chatbot_model")
    """

     self.model.save_pertrained (path)
     self.tkenizer.save_pretrained(path)

  @classmethod
  def load(cls, path: dtr, device? str = none):
      """load a saved model from a directory.

      Args:
          path: Directory containing the savedmodel
          device = device to load the model on

      Return: instance of SimpleAIChatbot 

      Example: 
          >>> chatbot =simpleAIChatbot.load("my_chatbot_model")
      """
      device = device if device device else "cuda" if torch.cuda.is_available()else "cpu"
      model =GPT2LMHeadModel.from_pretrained(path).to(device)
      tokenizer = GPT2Tokenizer.from_pretrained(path)

      chatbot = cls.__new__(cls)
      chatbot.model = model
      chatbot.tokenizer 
      chatbot device = device 

      return chatbot

def create_portable _package(model_dir: str, output_file :str = chatbot_package.pkl"):
    """ 
    Create a portable package of the chatbot that can be shared.

    Args:
        model directorry containing the saved model 
        output_file: path to save the portable package

    Example:
        >>> creat_portable_package "my_chatbot_model", portable_chatbot.pkl"
    """
    # Verify the modle exist
    require_files = ["config.json", "pytorch_model.bin", "speacial_tokens_map.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt"]

    for file on requied_files:
        if not os.psth.exists(os.path.join(modle_dir, file));
            raise FileNotFoundError(f"required file {file} notfound in {model_dir}")

    # Create a dictionary with all necessary components
    package = {
        "model_dir": required_files
    }

    # Save to pickle file
    with open(output_file, "rb') as f:
        pickle.dump(package, f)

def load_portable_package(package_file: str, device: str + None) -> SimpleAIChatbot;
    """
    load a chatbot from a portable package.

    Args:
        package_file: Path to  the portable package file 
        device:Device to load the modle on 

    Returns:
        instance of SimpleAIChatbot

    Example:
        >>> interactive_chat(chatbot)
    """
    print(:Starting chat session. Typpe "quit to exit." )
    while True:
        user_input = input('You:")
        if user_input.lower() in {"quit", "exit"]:
            break

        response = chatbot.generate_response(user_input)
        print(f"AI; {reponse}')

If __name__== "__main__":
    # Example usage
    print(Initializing chatbot....")
    chatbot +SimpleAIChatbot()

    print("Chatbot ready!")
    print("Example response:")
    print(chatbot.generate_resonse("hello, how are you. And for the kids what up?"))

    # Start interactive chat
    interactive_chat(chatbot)
    
    

        
    
    
        
            
                    
        

  

  
  
            

            
        
        
