# Autocorrect using N-gram Language Model and Levenshtein Distance

This project implements an autocorrect system using a combination of n-gram language models and Levenshtein distance.

## Installation

1. Ensure you have Python 3.7+ installed.
2. pip install -r requirements.txt

## Usage

To run the autocorrect model and generate predictions:
```angular2html
python main.py --gold_fn assignment_data/val_fold.csv --pred_fn val_predictions.csv --out_dir val_performances --out_pref your_model
```
To evaluate the model's performance:
```angular2html
python eval.py --gold_fn assignment_data/val_fold.csv --pred_fn val_predictions.csv --out_dir val_performances --out_pref your_model
```

## Files
```angular2html

- `main.py`: Main script for training the model and generating predictions
- `eval.py`: Evaluation script
- `requirements.txt`: List of required Python packages
- `val_predictions.csv`: Predictions for the validation set
- `test_predictions.csv`: Predictions for the test set
```
## Approach

This autocorrect system uses a bidirectional 3-gram language model combined with Levenshtein distance for candidate generation and scoring. It tokenizes input using NLTK's word_tokenize function and builds a vocabulary from the training data.

For more details on the implementation and performance, please refer to the accompanying report.

