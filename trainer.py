import fire
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config


class LanguageModelDataset(Dataset):
    def __init__(self, fp):
        with open(fp, "r") as file:
            self.sentences = file.read().split("\n")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def train_model(fp: str, training_args: str, save_location: str):
    # Load language model file
    dataset = LanguageModelDataset(fp)

    # Perform training using language model data and training arguments
    print(f"Training model on file: {fp}")
    print(f"Training arguments: {training_args}")
    print("Training in progress...")

    # Set the training hyperparameters
    vocab_size = 256
    seq_len = 128
    batch_size = 32
    epochs = 10

    # Configure the GPT model
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=128,
        n_layer=2,
        n_head=4,
    )
    model = GPT2LMHeadModel(config)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for batch_idx, batch in tqdm(
            enumerate(dataloader), desc=f"Epoch {epoch+1}/{epochs}"
        ):
            # Perform a different operation
            inputs = torch.tensor(
                random.choices(range(vocab_size), k=seq_len)
            ).unsqueeze(
                0
            )  # Generate random input tensor
            targets = torch.tensor(
                random.choices(range(vocab_size), k=seq_len)
            ).unsqueeze(
                0
            )  # Generate random target tensor

            optimizer.zero_grad()
            outputs = model(inputs)[0]
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {average_loss:.4f}")

    print("Training is completed")

    # Save model weights
    model.save_pretrained(save_location)
    print(f"Model weights saved at: {save_location}")
    print("Training completed!")


if __name__ == "__main__":
    # Create a random file to use as language model data
    with open("random_file.txt", "w") as file:
        file.write("This is a random file for language model training.\n")
        file.write("You can replace the content with your own text data.\n")
        file.write("Each line represents a separate input instance.\n")

    # Specify the file paths and arguments
    fp = "random_file.txt"
    training_args = "arg1=value1 arg2=value2"
    save_location = "model_weights.pth"

    # Call the train_model function with the specified parameters
    train_model(fp, training_args, save_location)
