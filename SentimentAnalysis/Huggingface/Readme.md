# Sentiment Analysis with Hugging Face Transformers

This project demonstrates a simple sentiment analysis pipeline using the Hugging Face Transformers library in a Google Colab environment. The code allows you to upload a text file, process the contents to analyze the sentiment of each sentence, and output the results.

## Overview

The main steps involved in the code are:
1. Installing necessary libraries.
2. Initializing a pre-trained sentiment analysis model.
3. Uploading a text file via Google Colab.
4. Processing the file to extract sentences.
5. Analyzing the sentiment of each sentence.
6. Displaying the sentiment and confidence score for each sentence.

## Requirements

- Python 3.x
- Google Colab (recommended environment) (https://colab.research.google.com/)
- Hugging Face Transformers library
- PyTorch

## Sample Data

The repo comes with a sample dataset that you can use to test the sentiment analysis. 

## Usage

### Clone or Download the Repository:

- Clone the repository to your local machine using `git clone` or download the ZIP file and extract it.

### Open the Code in Google Colab:

- Open the `sentiment_analysis.ipynb` file in Google Colab.

### Upload a Text File:

- Run the code cell that prompts you to upload a file. You can upload any text file from your local machine.
- The file's content should consist of sentences separated by commas.

### Process and Analyze Sentiments:

- The code will automatically process the uploaded text, split it into sentences, and analyze the sentiment of each sentence using the pre-trained model.

### Output:

- The sentiment (`POSITIVE`, `NEGATIVE`, or `NEUTRAL`) and the confidence score for each sentence will be printed in the Colab notebook.

## How It Works

### Sentiment Analysis Pipeline:
The Hugging Face Transformers library provides a `pipeline` function that simplifies the process of using pre-trained models. In this case, the `"sentiment-analysis"` pipeline is used, which leverages a model that is fine-tuned on a large dataset of text to classify the sentiment of given input sentences.

### File Upload:
Google Colab's `files.upload()` function is used to allow users to upload a file from their local machine. The uploaded file is processed to extract individual sentences for sentiment analysis.

### Output:
The result of the sentiment analysis is printed in the format of `label: <SENTIMENT>, with score: <CONFIDENCE>`. The confidence score is a float value between 0 and 1, indicating how confident the model is in its prediction.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to suggest improvements or report bugs.
