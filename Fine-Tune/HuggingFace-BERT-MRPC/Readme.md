# BERT Fine-Tuning on GLUE MRPC Dataset

This repository contains the code and resources to fine-tune a BERT model on the GLUE MRPC (Microsoft Research Paraphrase Corpus) dataset using PyTorch and the Hugging Face Transformers library. The fine-tuning process involves training the BERT model to classify whether two sentences are paraphrases or not.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Testing Custom Sentences](#testing-custom-sentences)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates how to fine-tune a pre-trained BERT model on the MRPC dataset. The MRPC dataset contains pairs of sentences, with labels indicating whether the sentences in each pair are semantically equivalent. The fine-tuning process adjusts the weights of the BERT model to perform well on this specific task.

## Installation

To run this project, you'll need to install the necessary Python libraries. You can install them using pip.
```bash
pip install transformers datasets torch

```

## Dataset

The MRPC dataset is part of the GLUE benchmark and can be easily loaded using the `datasets` library:

```python
from datasets import load_dataset
dataset = load_dataset("glue", "mrpc")
```

## Training the Model
To train the model, run the glue_mrpc_finetune.py script. This script loads the pre-trained BERT model, tokenizes the MRPC dataset, and fine-tunes the model.

```bash
python glue_mrpc_finetune.py
```
The training script will save the model checkpoints to the ./results directory.

## Evaluating the Model
After training, you can evaluate the model's performance on the validation set:

```python
from transformers import Trainer
# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("results")
trainer = Trainer(model=model, args=training_args, eval_dataset=tokenized_datasets["validation"])
# Evaluate
metrics = trainer.evaluate()
print(metrics)
```

##Testing Custom Sentences
You can test the fine-tuned model on custom sentence pairs to see if they are paraphrases:

```python
def test_sentence_pair(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure the inputs are on the same device as the model
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    print(f"Prediction: {'Paraphrase' if predictions == 1 else 'Not a paraphrase'}")

test_sentence_pair("The cat is on the mat.", "A cat is sitting on the mat.")
```
## Results
After training the model, you can expect to achieve a certain level of accuracy and other metrics on the validation set. The exact performance will depend on the training parameters and the dataset used.

## Contributing
If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. All contributions are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
