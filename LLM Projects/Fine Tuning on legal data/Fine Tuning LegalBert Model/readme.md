# Ensemble Fine-Tuning of LegalBERT for Indian Court Judgment Classification

## Project Overview
This project explores an advanced machine learning technique by leveraging ensemble learning to fine-tune multiple instances of the LegalBERT model on a dataset of Indian Supreme Court and High Court judgments. The primary objective is to enhance the predictive performance of the LegalBERT model for legal case outcomes by utilizing the ensemble approach, which is inspired by techniques commonly used in decision tree ensembles like Random Forests.

## Concept and Methodology

### Ensemble Learning
Ensemble learning involves training multiple models and combining their predictions to improve overall performance. In this project, six separate instances of the LegalBERT model are fine-tuned on the same legal dataset. Each model is trained independently, and their predictions are later combined to form an ensemble.

### LegalBERT
LegalBERT is a pre-trained language model specifically designed for legal text, which has been further fine-tuned on Indian legal judgments. This project utilizes the LegalBERT model as a base to create multiple fine-tuned models, each slightly different due to variations in training.

### Training Process
Each of the six LegalBERT models is trained using the same dataset, which consists of Indian court judgments. The training involves optimizing the models using the AdamW optimizer, with weight decay to prevent overfitting. Gradient clipping and a learning rate scheduler are also employed to enhance training stability.

### Ensemble Prediction
After the models are trained, their predictions are combined to form an ensemble. The ensemble's final prediction is typically made by averaging the outputs of all models or using a voting mechanism.

## Benefits of the Ensemble Approach

### Improved Accuracy and Robustness
By combining multiple models, the ensemble approach reduces the variance in predictions, leading to improved accuracy and robustness compared to individual models.

### Generalization
Ensembles often generalize better to unseen data, making them more reliable for real-world applications, such as legal case outcome prediction.

## Evaluation
The performance of the ensemble is evaluated using standard metrics such as accuracy, precision, recall, and F1 score. Additionally, a classification report and training loss curves for each model are generated, providing insights into the models' behavior during training and their contributions to the ensemble's performance.

## Visualization
The project includes visualizations of the training loss curves for each of the six models, allowing for a comparison of their convergence rates and stability during training. This helps in understanding which models performed better during the training process.

## Application
The resulting ensemble model is particularly suited for tasks such as legal research, analytics, and decision support, where accurate predictions of legal case outcomes are crucial. This ensemble model, fine-tuned specifically on Indian legal text, is expected to offer superior performance compared to a single LegalBERT model.

## References
This project is inspired by and references the [InLegalBERT model by Law and AI, IIT Kharagpur](https://huggingface.co/law-ai/InLegalBERT), which serves as the foundational model for fine-tuning in this work.

## Important Notes
- **Code Disclaimer:** The sample code provided in this repository may not work as expected due to updates to the Indian Kanoon website. The code and sample dataset are provided for reference only.
- **Private Project:** This project was developed for a private client, and only a sample dataset is shared in this repository.

---
**Author:** Maya Unnikrishnan 
**Date:** Project Completed on July 2024
