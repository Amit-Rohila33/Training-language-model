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

- git clone https://github.com/Amit-Rohila33/language-model-training.git

2. Change to the project directory:

- cd language-model-training


3. Create a text file for training data:

- Edit the file `random_file.txt` and replace the content with your own text data. Each line in the file represents a separate input instance.

4. Run the training script:

- python trainer.py


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
- save_location = "model_weights.pth"

2. Load the model with the saved weights
- model = GPT2LMHeadModel.from_pretrained(save_location)

Use the loaded model for inference or further training



# Language Model Inference with FastAPI

The **server.py** script provides a FastAPI server for generating text using the trained language model.

- Ensure you have trained the language model and saved the model weights in the specified location.

- In the root directory of the repository, you will find a file called server.py.

#### Run the FastAPI server:

- "uvicorn server:app --reload"
The server will start running on http://localhost:8000 or http://127.0.0.1:8000.

- To generate text, send a POST request to http://localhost:8000/generate with a JSON payload containing the text field.
For example:

POST http://localhost:8000/generate

Content-Type: application/json

{
  "text": "This is an example text."
}

The server will respond with the generated text in the response JSON, where the input text has undergone a complex random transformation.


# How to use the test.py

To use the stress testing script:

- Create the test.py file in the root directory of your repository and paste the provided code.
- Make sure your FastAPI server is running on http://localhost:8000.
- Run the test.py script using python test.py.
- The script will create multiple threads and send simultaneous requests to the FastAPI server's /generate endpoint.
- Each thread will print the JSON response received from the server.
