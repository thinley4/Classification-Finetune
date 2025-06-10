# Spam Detection using GPT-2 Finetuning

This project demonstrates how to finetune a GPT-2 model for spam text message classification. The implementation follows the approach described in Sebastian Raschka's book "Build a Large Language Model From Scratch".

## Overview

This project demonstrates how to take a pretrained GPT-2 model and finetune it for a binary classification task (spam vs. non-spam messages). It includes:

- Loading and preprocessing the SMS Spam Collection dataset
- Adding a classification head to the GPT-2 model
- Finetuning the model for classification
- Evaluating model performance
- Deploying the model via a Chainlit chat interface

## Dataset

The SMS Spam Collection dataset consists of labeled text messages categorized as spam or ham (not spam). The preprocessing pipeline:
1. Balances the dataset to address class imbalance
2. Converts string labels to integers (ham → 0, spam → 1) 
3. Splits data into training, validation, and test sets
4. Tokenizes and pads sequences to handle variable text lengths

## Model Architecture

The project uses a pretrained GPT-2 model with the following modifications:
- The original language modeling head is replaced with a binary classification head
- The last transformer block and final layer normalization are unfrozen for finetuning
- The model learns to classify messages based on the last token's representation

## Requirements

```
torch
tiktoken
pandas
numpy
matplotlib
chainlit
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Model Training
- Data preprocessing and loading
- Model configuration and initialization
- Training loop with loss and accuracy tracking
- Model evaluation and saving

### Inference API
Run the Chainlit app to use the model through a chat interface:

```bash
chainlit run app.py --port 8001
```

Use port 8001 to avoid conflicts with other services.

## Model Performance

After finetuning, the model achieves:
- Training accuracy: ~98%
- Validation accuracy: ~95%
- Test accuracy: ~93%

## Project Structure

```
.
├── app.py                    # Chainlit web interface
├── chainlit.md               # Chainlit configuration
├── gpt2-model.pth            # Pretrained GPT-2 model weights
├── main.ipynb                # Main notebook with training code
├── README.md                 # This file
├── requirements.txt          # Project dependencies
├── review_classifier.pth     # Finetuned classification model
├── utils.py                  # Utility functions
└── datasets/                 # Dataset folders
```

## Acknowledgments

This project is based on concepts from Sebastian Raschka's book "Build a Large Language Model From Scratch" and uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).