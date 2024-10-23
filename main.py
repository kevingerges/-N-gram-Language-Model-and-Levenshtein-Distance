import pandas as pd
import nltk
import ssl
import Levenshtein
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import math
import argparse

# ignore SSL certificate errors
# just saw this online and it fixed my import error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')

# load training data
train_df = pd.read_csv('assignment_data/train_fold.csv')
val_df = pd.read_csv('assignment_data/val_fold.csv')
test_df = pd.read_csv('assignment_data/test_fold_input_only.csv')

print("Train DataFrame columns:", train_df.columns)
print("Validation DataFrame columns:", val_df.columns)
print("Test DataFrame columns:", test_df.columns)

def tokenize_text(text):
    # i wanna tokenize the text here
    if not isinstance(text, str) or text.strip() == '':
        return []
    return word_tokenize(text.lower())

def build_vocab(tokenized_texts, min_freq=1):
    # ima build the vocab here
    word_counts = Counter()
    for tokens in tokenized_texts:
        word_counts.update(tokens)
    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    return vocab

def extract_ngrams(tokens, n):
    # extract n-grams from tokens
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return ngrams

def build_ngram_counts(tokenized_texts, n):
    # count them n-grams
    ngram_counts = defaultdict(int)
    for tokens in tokenized_texts:
        ngrams = extract_ngrams(tokens, n)
        for ngram in ngrams:
            ngram_counts[ngram] += 1
    return ngram_counts

def calculate_ngram_probabilities(ngram_counts, n_minus_one_counts, vocab_size, alpha=1):
    # calculate them probabilities
    ngram_probabilities = {
        ngram: (count + alpha) / (n_minus_one_counts.get(ngram[:-1], 0) + alpha * vocab_size)
        for ngram, count in ngram_counts.items()
    }
    return ngram_probabilities

def reverse_tokens(tokenized_texts):
    # flip them tokens
    return [tokens[::-1] for tokens in tokenized_texts]

def generate_candidates(word, vocab, max_distance=2):
    # generate some word candidates
    candidates = [(vocab_word, Levenshtein.distance(word, vocab_word))
                  for vocab_word in vocab
                  if abs(len(vocab_word) - len(word)) <= max_distance and
                  Levenshtein.distance(word, vocab_word) <= max_distance]
    return candidates

def correct_message(tokens, forward_ngram_probs, backward_ngram_probs, vocab, n, weight=0.5):
    # correct the message yo
    corrected_tokens = []
    for i, word in enumerate(tokens):
        if word in vocab:
            corrected_tokens.append(word)
            continue

        # get that context
        left_context = tokens[max(0, i - (n - 1)): i]
        forward_context = tuple(left_context)
        right_context = tokens[i + 1: i + n]
        backward_context = tuple(right_context[::-1])

        candidates = generate_candidates(word, vocab)

        # score them candidates
        candidate_scores = []
        for candidate_word, distance in candidates:
            forward_ngram = forward_context + (candidate_word,)
            forward_lm_prob = forward_ngram_probs.get(forward_ngram, 1e-6)
            backward_ngram = (candidate_word,) + backward_context
            backward_lm_prob = backward_ngram_probs.get(backward_ngram, 1e-6)

            lm_prob = (math.log(forward_lm_prob) + math.log(backward_lm_prob)) / 2
            lev_similarity = 1 - (distance / max(len(candidate_word), len(word)))

            combined_score = weight * lm_prob + (1 - weight) * lev_similarity
            candidate_scores.append((candidate_word, combined_score))

        # pick the best one
        best_candidate = max(candidate_scores, key=lambda x: x[1])[0] if candidate_scores else word
        corrected_tokens.append(best_candidate)

    return corrected_tokens

def main(args):
    # start tokenizing training messages
    train_tokens = [tokenize_text(text) for text in train_df['gold_msg']]
    vocab = build_vocab(train_tokens, min_freq=2)

    # then build forward n-gram model
    n = 3
    forward_ngram_counts = build_ngram_counts(train_tokens, n)
    forward_n_minus_one_counts = build_ngram_counts(train_tokens, n - 1)
    vocab_size = len(vocab)
    forward_ngram_probs = calculate_ngram_probabilities(
        forward_ngram_counts, forward_n_minus_one_counts, vocab_size, alpha=1)

    # also build backward n-gram model
    backward_train_tokens = reverse_tokens(train_tokens)
    backward_ngram_counts = build_ngram_counts(backward_train_tokens, n)
    backward_n_minus_one_counts = build_ngram_counts(backward_train_tokens, n - 1)
    backward_ngram_probs = calculate_ngram_probabilities(
        backward_ngram_counts, backward_n_minus_one_counts, vocab_size, alpha=1)

    # get the correct validation messages
    val_input_tokens = [tokenize_text(text) for text in val_df['corrupt_msg']]
    val_corrected_tokens = []
    for tokens in val_input_tokens:
        corrected_tokens = correct_message(tokens, forward_ngram_probs, backward_ngram_probs, vocab, n, weight=0.5)
        val_corrected_tokens.append(' '.join(corrected_tokens))

    # saving validation predictions
    val_predictions = pd.DataFrame({
        'id': val_df['Unnamed: 0'],
        'pred_msg': val_corrected_tokens
    })
    val_predictions.to_csv('val_predictions.csv', index=False)

    # get the correct test messages
    test_input_tokens = [tokenize_text(text) for text in test_df['corrupt_msg']]
    test_corrected_tokens = []
    for tokens in test_input_tokens:
        corrected_tokens = correct_message(tokens, forward_ngram_probs, backward_ngram_probs, vocab, n, weight=0.5)
        test_corrected_tokens.append(' '.join(corrected_tokens))

    # saving test predictions
    test_predictions = pd.DataFrame({
        'id': test_df['Unnamed: 0'],
        'pred_msg': test_corrected_tokens
    })
    test_predictions.to_csv('test_predictions.csv', index=False)

    print("Predictions generated and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_fn", type=str, help="Disc location of CSV with gold messages", required=True)
    parser.add_argument("--pred_fn", type=str, help="Disc location of CSV with predicted messages", required=True)
    parser.add_argument("--out_dir", type=str, help="Directory to write performance files", required=True)
    parser.add_argument("--out_pref", type=str, help="File prefix for output files", required=True)
    parser.add_argument("--ostrich_summary_perf_fn", type=str, help="Location of ostrich alg summary perf", required=False)

    args = parser.parse_args()
    main(args)