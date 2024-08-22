# Comprehensive NLP with Transformer and LLM Models for Summarization

## Overview
This project demonstrates the use of pre-trained BERT and T5 models for text summarization tasks. It includes examples of text summarization using BERT, text summarization using T5, and evaluation of summarization quality using METEOR and ROUGE metrics.

## Text Report Summarization Using BERT: Tokenization, Embedding, and Importance Scoring

### Steps:

1. **Load BERT Model and Tokenizer**:
   - The script initializes a pre-trained BERT model and its corresponding tokenizer using the `bert-base-uncased` configuration.

2. **Read Text Data**:
   - It reads lines from a file named `text.txt`, where each line represents a separate text report or document.

3. **Tokenization**:
   - Each text report is tokenized into BERT-compatible tokens, including special tokens like `[CLS]` and `[SEP]`.

4. **Padding**:
   - Tokenized sequences are padded to ensure all sequences have the same length, based on the longest sequence in the dataset.

5. **Create Tensors**:
   - Converts the padded token sequences into PyTorch tensors for model input.

6. **Attention Masks**:
   - Generates attention masks to distinguish between actual tokens and padding tokens.

7. **Generate Embeddings**:
   - Computes embeddings for each token in the text reports using the BERT model. Each text report is mapped to its embeddings.

8. **Calculate Sentence Scores**:
   - Computes a simple importance score for each text report by summing the values of its embeddings.

9. **Select Top Sentences**:
   - Identifies the top text report(s) with the highest importance scores to form a summary.

10. **Output Summary**:
    - Prints the selected text report(s) as the summary.

## Text Summarization with T5: Training, Fine-Tuning, and Generating Summaries

### Steps:

1. **Load Pre-trained T5 Model and Tokenizer**:
   - Initializes the T5 model and tokenizer using the `t5-small` configuration for text summarization tasks.

2. **Read Data**:
   - Reads text data from `text.txt`, where each line represents a text document or report to be summarized.

3. **Define Custom Dataset Class**:
   - Creates a `SummarizationDataset` class that preprocesses text data by tokenizing it and preparing it for model input. It includes handling padding and truncation to fit the T5 model's requirements.

4. **Initialize Dataset and DataLoader**:
   - Sets up the dataset and DataLoader for batching and shuffling the data during training. The maximum source length for the text is set to 128 tokens.

5. **Define Optimizer and Loss Function**:
   - Uses the AdamW optimizer with a learning rate of 1e-5 for fine-tuning the T5 model. The loss function is CrossEntropyLoss, which helps in training the model to predict the correct summary tokens.

6. **Training Loop**:
   - Trains the T5 model over 5 epochs, updating the model parameters based on the loss computed from the text data.

7. **Summarization Function**:
   - Defines a function `generate_summary` to generate summaries for given input text by encoding the input, generating summary tokens, and decoding them back into text.

8. **Test Summarization Function**:
   - Tests the summarization function with an example input text and prints the generated summary.
