# Web Scraping and Preprocessing of Indian Court Judgments for LegalBERT Fine-Tuning

## Project Overview
This project focuses on the web scraping, labeling, and preprocessing of Indian Supreme Court and High Court judgments from the Indian Kanoon website. The primary objective is to extract, categorize, and preprocess the data to create a structured dataset that can be used for further analysis and to fine-tune the LegalBERT language model for specific legal tasks, such as predicting case outcomes.

## Use Case
The preprocessed data serves as valuable input for fine-tuning LegalBERT, a specialized language model tailored for legal text. By training LegalBERT on this dataset, we aim to enhance its ability to understand and predict outcomes of legal cases, thereby making it a powerful tool for legal research, analytics, and decision support.

## Project Phases

### Phase 1: Web Scraping and Data Extraction
**Objective:** Scrape all available data (judgments) from the Indian Kanoon website and save it in a structured format.

**Methodology:**

- **Web Scraping:** The project begins by scraping data from the Indian Kanoon website, specifically targeting the titles, links, and facts (case details) of the judgments.
- **Data Storage:** The scraped data is stored in a CSV file with the following columns: `Title`, `Link`, and `Fact`.

### Phase 2: Labeling the Data
**Objective:** Assign labels to the scraped data, categorizing each case as either "Accepted" or "Rejected" for use in training the LegalBERT model.

**Methodology:**

- **Dictionary Mapping:** A dictionary (`key_to_label`) is created to map various case outcomes to standardized labels (`ACCEPTED`, `REJECTED`, and other ranks if needed).
- **Label Assignment:** Each row in the dataframe is analyzed using a case-insensitive search to find keywords in the `Fact` column, which are then mapped to the appropriate labels.
- **Binary Labeling:** A new column `Label` is created, where cases labeled as "Rejected" are assigned `0`, and those labeled as "Accepted" are assigned `1`.

### Phase 3: Data Preprocessing for LegalBERT Fine-Tuning
**Objective:** Refine the labeling process by identifying specific outcomes within the text, ensuring that the data is well-prepared for fine-tuning the LegalBERT model.

**Methodology:**

- **Regular Expressions (Regex):** Use regex patterns to identify specific phrases within the `Fact` column that indicate case outcomes (e.g., "appeal accepted," "petition rejected").
- **Outcome Extraction:** When an outcome phrase is detected, it is extracted and used to populate a new column `PreprocessedOutcome`. Corresponding labels are stored in another new column, `Label2`.

## Conclusion
The preprocessed dataset, enriched with additional columns such as `Result`, `Label`, `PreprocessedOutcome`, and `Label2`, provides a more detailed understanding of case outcomes. This dataset is particularly well-suited for fine-tuning LegalBERT, enabling the model to better understand and predict legal case outcomes based on textual input.

## Important Notes
- **Code Disclaimer:** The sample code provided in this repository may not work as expected due to updates to the Indian Kanoon website. The code and sample dataset are provided for reference only.
- **Private Project:** This project was developed for a private client, and only a sample dataset is shared in this repository.

## Reference
For fine-tuning the LegalBERT model, we referred to the [InLegalBERT model by Law and AI, IIT Kharagpur](https://huggingface.co/law-ai/InLegalBERT). This model is a version of LegalBERT tailored specifically for Indian legal texts and serves as an excellent foundation for further fine-tuning using the preprocessed data from this project.

---

**Author:** Maya Unnikrishnan 
**Date:** Project Completed on July 2024
