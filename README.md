# Financial Sentiment Analysis Project

## Goal
This project studies financial sentiment classification using two datasets with:
- in-domain evaluation
- cross-dataset evaluation
- baseline models
- transformer fine-tuning
- optional domain-adaptive pretraining (DAPT)

## Project Structure
- `src/data`: data loading, cleaning, label mapping, splitting
- `src/models`: baseline and transformer training
- `src/evaluation`: metrics, confusion matrix, error analysis
- `configs`: experiment configs
- `outputs`: saved metrics, predictions, figures, checkpoints

## Planned Models
- TF-IDF + Logistic Regression
- FinBERT / BERT-based fine-tuning
- Optional DAPT

## Run Strategy
Code is developed locally and executed on cloud environments for training.