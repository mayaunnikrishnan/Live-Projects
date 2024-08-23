# Sentiment Analysis and Chatbot for Indian E-Commerce Platform

## Project Overview

This project involves fine-tuning the BERT (Bidirectional Encoder Representations from Transformers) base model (`bert-base-uncased`) for sentiment analysis on a dataset of Indian e-commerce reviews, specifically from Amazon product purchase reviews. The primary objective is to utilize sentiment analysis to develop a customized chatbot for an Indian e-commerce platform. The chatbot interacts with customers based on the sentiment expressed in their reviews, collects detailed ratings for various product features, and gathers suggestions for improvements. Additionally, the project includes the development of an admin dashboard that visualizes customer ratings and feedback to assist in making data-driven decisions for service enhancements.

## Concept and Methodology

### Fine-Tuning BERT for Sentiment Analysis

The BERT model, pre-trained on a large corpus of English text, was fine-tuned on a dataset of Indian product reviews to classify the sentiment into three categories: **Positive**, **Negative**, and **Neutral**. The fine-tuning process involved the following steps:

#### Tokenization and Preprocessing

- Reviews were tokenized and preprocessed using the BERT tokenizer (`BertTokenizer.from_pretrained('bert-base-uncased')`).
- Text was encoded into token IDs, padded to a maximum length of 128 tokens, and converted to PyTorch tensors.

#### Model Training

- The pre-trained BERT model (`BertForSequenceClassification`) was fine-tuned on the review data using a PyTorch training loop.
- The model was trained for 5 epochs, with the optimizer's gradients being reset at the end of each batch.
- Loss was computed and backpropagated to update the model's weights, optimizing it for sentiment classification.

#### Evaluation

- Post-training, the model's performance was evaluated based on accuracy, a classification report, and a confusion matrix, providing insights into how well the model categorized sentiments.

### Chatbot Development

The fine-tuned sentiment analysis model was integrated into a customized chatbot designed for an Indian e-commerce platform. The chatbot performs the following functions:

#### Sentiment Detection

- The chatbot analyzes the sentiment of the customer's review using the fine-tuned BERT model.
- Based on the sentiment, the chatbot adapts its responses to either thank the customer for positive feedback, address concerns in negative feedback, or probe further in the case of neutral feedback.

#### Feature Rating Collection

- The chatbot engages the customer to rate various product features (e.g., quality, value for money, delivery experience) on a scale.
- This detailed rating helps in understanding specific areas that need improvement.

#### Suggestion Gathering

- The chatbot prompts the customer to provide suggestions for improving products or services, offering a direct line of feedback.

### Admin Dashboard Development

To help the e-commerce site's administrators make informed decisions, an admin dashboard was developed using a Flask web application. The dashboard features:

#### Secure Access

- The dashboard is accessible only to authorized personnel through a password-protected login.

#### Real-Time Visualization

- The dashboard visualizes customer ratings and sentiment data using graphs and charts, providing a clear overview of current customer satisfaction levels and areas needing attention.
- Key metrics and trends are displayed to aid in strategic decision-making.

## Application

This project is particularly useful for enhancing customer engagement on e-commerce platforms. The sentiment analysis-powered chatbot improves customer interaction by addressing issues proactively and gathering detailed feedback. The admin dashboard offers a centralized view of customer sentiments and ratings, enabling the business to continuously improve its offerings based on direct customer input.

## Conclusion

By fine-tuning BERT on Indian e-commerce data and integrating it into a customer-facing chatbot and an admin dashboard, this project demonstrates how AI can be applied to improve customer service and product quality in a targeted manner. The integration of sentiment analysis with interactive customer service tools provides a comprehensive approach to understanding and responding to customer needs in real-time.
note - (i didn't uploaded any model folder in flask app ,because fine tuned model is a large file)
## Author

**Maya Unnikrishnan**

## Date

**Project Completed on July 2024**
