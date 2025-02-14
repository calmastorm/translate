from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import os
from config import *

def download(model_name=PRETRAINED_MODEL_NAME, save_directory=PRETRAINED_SAVE_DIRECTORY):
    print(f'Downloading {model_name} to {save_directory}')

    os.makedirs(save_directory, exist_ok=True)

    model_file = os.path.join(save_directory, "pytorch_model.bin")
    tokenizer_file = os.path.join(save_directory, "spiece.model")

    # download tokenizer
    if os.path.exists(tokenizer_file):
        print("Tokenizer already exists, skipping download.")
    else:
        print("Downloading tokenizer...")
        tokenizer = MT5Tokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_directory)

    # download model
    if os.path.exists(model_file):
        print("Model already exists, skipping download.")
    else:
        print("Downloading model...")
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(save_directory)

    print(f"Model {model_name} has been downloaded and saved at {save_directory}")

if __name__ == '__main__':
    download()