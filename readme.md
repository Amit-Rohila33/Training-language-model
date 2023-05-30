# Language Model Training with GPT-2

This repository contains a script for training a language model using GPT-2 architecture. The script takes a text file as input, performs training on the language model using the provided data, and saves the trained model weights for future use.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python 3
- PyTorch
- Transformers
- tqdm

You can install the dependencies by running the following command:

pip install torch transformers tqdm

## Usage

1. Clone the repository:

git clone https://github.com/Amit-Rohila33/language-model-training.git

2. Change to the project directory:

cd language-model-training


3. Create a text file for training data:

Edit the file `random_file.txt` and replace the content with your own text data. Each line in the file represents a separate input instance.

4. Run the training script:

python trainer.py


The script will train the language model on the provided data. The training progress will be displayed, showing the average loss after each epoch. Once training is completed, the model weights will be saved in the file specified by `save_location`.

## Customization

- You can modify the hyperparameters of the training process by editing the values in the `train_model` function.
- The GPT-2 model architecture can be customized by adjusting the configuration parameters in the `GPT2Config` initialization.
- Additional command-line arguments can be added to the `train_model.py` script using the `fire` library.

## Acknowledgements

This project uses the `GPT2LMHeadModel` and `GPT2Config` classes from the Hugging Face Transformers library. For more information about GPT-2 and Transformers, refer to the official documentation.

## To Load the models write the following codes

from transformers import GPT2LMHeadModel

1. Specify the path to the saved model weights
save_location = "model_weights.pth"

2. Load the model with the saved weights
model = GPT2LMHeadModel.from_pretrained(save_location)

Use the loaded model for inference or further training



