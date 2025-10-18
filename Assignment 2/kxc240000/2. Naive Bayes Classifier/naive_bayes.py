# ------------------------------------------------------------------ Big Data Assignment 2 ----------------------------------------------------------------- #
# Name - Kushal Choudhary
# Net ID - kxc240000
# --------------------------------------------------------- Naive Bayes Classifier using MapReduce --------------------------------------------------------- #

import argparse
import os
import urllib.request
import zipfile
import math
from pyspark.sql import SparkSession
from collections import defaultdict
import re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Dataset URL")
    p.add_argument("--test-split", type=float, default=0.2, help="Test set proportion")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def download_dataset(url):
    """Download and extract dataset"""
    zip_file = "dataset.zip"
    data_file = "SMSSpamCollection"
    
    if not os.path.exists(data_file):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, zip_file)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(zip_file)
        print("Dataset downloaded and extracted!")
    else:
        print("Using cached dataset")
    
    return data_file

def preprocess_text(text):
    """Tokenize, lowercase, remove punctuation, stop words"""
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = text.split()
    
    # Stop word removal (basic list)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'this',
                  'that', 'it', 'i', 'you', 'he', 'she', 'we', 'they'}
    
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    return tokens

if __name__ == "__main__":
    args = parse_args()
    
    # Download dataset
    data_file = download_dataset(args.input)
    
    print("\nStarting Spark...")
    spark = (SparkSession.builder
             .appName("NaiveBayesClassifier")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    print("Loading and preprocessing data...")
    
    # Load data
    lines = sc.textFile(data_file)
    
    # Parse: each line is "label\tmessage"
    def parse_line(line):
        parts = line.split('\t', 1)
        if len(parts) == 2:
            label = parts[0].strip()
            text = parts[1].strip()
            tokens = preprocess_text(text)
            return (label, tokens)
        return None
    
    data = lines.map(parse_line).filter(lambda x: x is not None)
    
    # Split train/test
    train_data, test_data = data.randomSplit([1 - args.test_split, args.test_split], seed=args.seed)
    train_data.cache()
    test_data.cache()
    
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"Training samples: {train_count}, Test samples: {test_count}")
    
    print("\n" + "="*60)
    print("TRAINING NAIVE BAYES MODEL")
    print("="*60)
    
    # Step 1: Calculate class priors P(class)
    print("\n1. Computing class priors...")
    class_counts = train_data.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collect()
    
    total_docs = sum(count for _, count in class_counts)
    class_priors = {label: count / total_docs for label, count in class_counts}
    
    print("Class Priors:")
    for label, prior in sorted(class_priors.items()):
        print(f"  P({label}) = {prior:.4f} ({int(prior * total_docs)} documents)")
    
    # Step 2: Calculate word likelihoods P(word|class) using MapReduce
    print("\n2. Computing word likelihoods using MapReduce...")
    
    # Map: emit (class, word) -> 1 for each word in each document
    word_class_pairs = train_data.flatMap(
        lambda x: [((x[0], word), 1) for word in x[1]]
    )
    
    # Reduce: count occurrences of each word in each class
    word_class_counts = word_class_pairs.reduceByKey(lambda a, b: a + b)
    
    # Get vocabulary
    vocabulary = train_data.flatMap(lambda x: x[1]).distinct().collect()
    vocab_size = len(vocabulary)
    print(f"  Vocabulary size: {vocab_size} unique words")
    
    # Count total words per class
    class_word_totals = train_data.flatMap(
        lambda x: [(x[0], len(x[1]))]
    ).reduceByKey(lambda a, b: a + b).collectAsMap()
    
    print("  Words per class:")
    for label, total in sorted(class_word_totals.items()):
        print(f"    {label}: {total} words")
    
    # Broadcast for efficient access
    class_priors_bc = sc.broadcast(class_priors)
    class_word_totals_bc = sc.broadcast(class_word_totals)
    vocab_size_bc = sc.broadcast(vocab_size)
    
    # Collect word counts for prediction
    word_counts_dict = defaultdict(lambda: defaultdict(int))
    for (label, word), count in word_class_counts.collect():
        word_counts_dict[label][word] = count
    
    word_counts_bc = sc.broadcast(dict(word_counts_dict))
    
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    # Step 3: Prediction function
    def predict(tokens):
        """Predict class using Naive Bayes formula with Laplace smoothing"""
        scores = {}
        
        for label in class_priors_bc.value.keys():
            # Start with log prior
            score = math.log(class_priors_bc.value[label])
            
            # Add log likelihoods for each word
            for word in tokens:
                word_count = word_counts_bc.value[label].get(word, 0)
                # Laplace smoothing: (count + 1) / (total + vocab_size)
                likelihood = (word_count + 1) / (class_word_totals_bc.value[label] + vocab_size_bc.value)
                score += math.log(likelihood)
            
            scores[label] = score
        
        # Return class with highest score
        return max(scores, key=scores.get)
    
    # Step 4: Make predictions
    predictions = test_data.map(lambda x: (x[0], predict(x[1])))
    
    # Step 5: Evaluate
    correct = predictions.filter(lambda x: x[0] == x[1]).count()
    accuracy = correct / test_count
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total test samples: {test_count}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    
    confusion = predictions.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a, b: a + b).collect()
    confusion_dict = {k: v for k, v in confusion}
    
    labels = sorted(class_priors.keys())
    print(f"\n{'Actual/Predicted':<15}", end="")
    for label in labels:
        print(f"{label:<15}", end="")
    print()
    
    for actual in labels:
        print(f"{actual:<15}", end="")
        for predicted in labels:
            count = confusion_dict.get((actual, predicted), 0)
            print(f"{count:<15}", end="")
        print()
    
    # Per-class metrics
    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    
    for label in labels:
        tp = confusion_dict.get((label, label), 0)
        fp = sum(confusion_dict.get((other, label), 0) for other in labels if other != label)
        fn = sum(confusion_dict.get((label, other), 0) for other in labels if other != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass: {label}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    print(f"\n{'='*60}\n")
    
    spark.stop()

# Input:
    # spark-submit naive_bayes.py --input https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip --test-split 0.2 --seed 42

# Output:
    # ============================================================
    # TRAINING NAIVE BAYES MODEL
    # ============================================================

    # 1. Computing class priors...
    # /Users/kushalc/spark/python/lib/pyspark.zip/pyspark/shuffle.py:65: UserWarning: Please install psutil to have better support with spilling
    # /Users/kushalc/spark/python/lib/pyspark.zip/pyspark/shuffle.py:65: UserWarning: Please install psutil to have better support with spilling
    # Class Priors:
    # P(ham) = 0.8609 (3838 documents)
    # P(spam) = 0.1391 (620 documents)

    # 2. Computing word likelihoods using MapReduce...
    # Vocabulary size: 7356 unique words
    # Words per class:
    #     ham: 33690 words
    #     spam: 9345 words

    # ============================================================
    # TESTING MODEL
    # ============================================================

    # ============================================================
    # RESULTS
    # ============================================================
    # Total test samples: 1116
    # Correct predictions: 1078
    # Accuracy: 0.9659 (96.59%)

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    # Actual/Predictedham            spam           
    # ham            963            26             
    # spam           12             115            

    # ============================================================
    # PER-CLASS METRICS
    # ============================================================

    # Class: ham
    # Precision: 0.9877
    # Recall: 0.9737
    # F1-Score: 0.9807

    # Class: spam
    # Precision: 0.8156
    # Recall: 0.9055
    # F1-Score: 0.8582

    # ============================================================