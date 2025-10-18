# ------------------------------------------------------------------ Big Data Assignment 2 ----------------------------------------------------------------- #
# Name - Kushal Choudhary
# Net ID - kxc240000
# --------------------------------------------------------- Naive Bayes Classifier using MapReduce --------------------------------------------------------- #

import argparse
import os
import urllib.request
import tarfile
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
    """Download and extract 20 newsgroups dataset"""
    tar_file = "20news.tar.gz"
    
    if not os.path.exists(tar_file):
        print(f"Downloading 20 Newsgroups dataset...")
        print("This may take 1-2 minutes...")
        urllib.request.urlretrieve(url, tar_file)
        print("Download complete!")
    else:
        print("Using cached dataset file")
    
    print("Extracting dataset...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(".")
    
    print("Extraction complete!")

def load_newsgroups_from_dir(directory):
    """Recursively load all text files from directory"""
    documents = []
    
    print(f"Scanning directory: {directory}")
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory {directory} does not exist!")
        return documents
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        # Get category from directory name
        category = os.path.basename(root)
        
        # Skip if this is the root directory or if no files
        if root == directory or len(files) == 0:
            continue
        
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 50:  # Skip very short files
                        documents.append((category, content))
            except Exception as e:
                pass
    
    print(f"  Loaded {len(documents)} documents from {directory}")
    return documents

def preprocess_text(text):
    """Enhanced preprocessing for newsgroups"""
    # Remove email headers
    lines = text.split('\n')
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '' and i > 5:  # Skip headers
            content_start = i
            break
    text = '\n'.join(lines[content_start:])
    
    # Remove email addresses and URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Lowercase and keep only letters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenization
    tokens = text.split()
    
    # Stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'this',
        'that', 'it', 'you', 'he', 'she', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'can', 'will', 'just', 'should', 'would', 'could', 'been', 'has',
        'have', 'had', 'do', 'does', 'did', 'am', 'one', 'two'
    }
    
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    return tokens

if __name__ == "__main__":
    args = parse_args()
    
    # Download dataset
    download_dataset(args.input)
    
    print("\nLoading documents...")
    
    # Load from both train and test directories (they're in current directory now)
    all_docs = []
    
    train_dir = "20news-bydate-train"
    if os.path.exists(train_dir):
        train_docs = load_newsgroups_from_dir(train_dir)
        all_docs.extend(train_docs)
    
    test_dir = "20news-bydate-test"
    if os.path.exists(test_dir):
        test_docs = load_newsgroups_from_dir(test_dir)
        all_docs.extend(test_docs)
    
    if len(all_docs) == 0:
        print("\nERROR: No documents loaded!")
        print("Checking current directory...")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  Directory: {item}")
                if '20news' in item:
                    # Try to show subdirectories
                    subdirs = os.listdir(item)[:5]
                    print(f"    Contains: {subdirs}")
        exit(1)
    
    print(f"\nTotal documents loaded: {len(all_docs)}")
    
    print("\nStarting Spark...")
    spark = (SparkSession.builder
             .appName("NaiveBayesClassifier_20News")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    # Create RDD
    data_rdd = sc.parallelize(all_docs)
    
    # Preprocess
    print("Preprocessing text...")
    processed_data = data_rdd.map(lambda x: (x[0], preprocess_text(x[1]))).filter(lambda x: len(x[1]) > 5)
    processed_data.cache()
    
    total_docs = processed_data.count()
    print(f"Documents after preprocessing: {total_docs}")
    
    # Split train/test
    train_data, test_data = processed_data.randomSplit([1 - args.test_split, args.test_split], seed=args.seed)
    train_data.cache()
    test_data.cache()
    
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"Training samples: {train_count}, Test samples: {test_count}")
    
    print("\n" + "="*60)
    print("TRAINING NAIVE BAYES MODEL")
    print("="*60)
    
    # Calculate class priors
    print("\n1. Computing class priors...")
    class_counts = train_data.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collect()
    
    total_train_docs = sum(count for _, count in class_counts)
    class_priors = {label: count / total_train_docs for label, count in class_counts}
    
    print(f"\nFound {len(class_priors)} categories:")
    for label, prior in sorted(class_priors.items(), key=lambda x: -x[1])[:10]:
        print(f"  {label:<30} P={prior:.4f} ({int(prior * total_train_docs)} docs)")
    if len(class_priors) > 10:
        print(f"  ... and {len(class_priors) - 10} more categories")
    
    # Calculate word likelihoods
    print("\n2. Computing word likelihoods using MapReduce...")
    
    word_class_pairs = train_data.flatMap(
        lambda x: [((x[0], word), 1) for word in x[1]]
    )
    
    word_class_counts = word_class_pairs.reduceByKey(lambda a, b: a + b)
    
    vocabulary = train_data.flatMap(lambda x: x[1]).distinct().collect()
    vocab_size = len(vocabulary)
    print(f"  Vocabulary size: {vocab_size:,} unique words")
    
    class_word_totals = train_data.flatMap(
        lambda x: [(x[0], len(x[1]))]
    ).reduceByKey(lambda a, b: a + b).collectAsMap()
    
    print(f"  Total words: {sum(class_word_totals.values()):,}")
    
    # Broadcast
    class_priors_bc = sc.broadcast(class_priors)
    class_word_totals_bc = sc.broadcast(class_word_totals)
    vocab_size_bc = sc.broadcast(vocab_size)
    
    word_counts_dict = defaultdict(lambda: defaultdict(int))
    for (label, word), count in word_class_counts.collect():
        word_counts_dict[label][word] = count
    
    word_counts_bc = sc.broadcast(dict(word_counts_dict))
    
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    def predict(tokens):
        scores = {}
        for label in class_priors_bc.value.keys():
            score = math.log(class_priors_bc.value[label])
            for word in tokens:
                word_count = word_counts_bc.value[label].get(word, 0)
                likelihood = (word_count + 1) / (class_word_totals_bc.value[label] + vocab_size_bc.value)
                score += math.log(likelihood)
            scores[label] = score
        return max(scores, key=scores.get)
    
    predictions = test_data.map(lambda x: (x[0], predict(x[1])))
    
    correct = predictions.filter(lambda x: x[0] == x[1]).count()
    accuracy = correct / test_count
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Test samples: {test_count}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class performance
    print(f"\n{'='*60}")
    print("TOP 10 CLASSES BY ACCURACY")
    print(f"{'='*60}")
    
    class_perf = predictions.map(lambda x: (x[0], (1 if x[0] == x[1] else 0, 1))).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    ).collect()
    
    for category, (correct_count, total) in sorted(class_perf, key=lambda x: -x[1][0]/x[1][1])[:10]:
        acc = correct_count / total
        print(f"{category:<30} {acc:>6.2%}  ({correct_count}/{total})")
    
    print(f"\n{'='*60}\n")
    
    spark.stop()

# Input:
    # spark-submit naive_bayes.py --input http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz --test-split 0.2 --seed 42

# Output:
    # Documents after preprocessing: 17463
    # Training samples: 13887, Test samples: 3576

    # ============================================================
    # TRAINING NAIVE BAYES MODEL
    # ============================================================

    # 1. Computing class priors...

    # Found 20 categories:
    # sci.crypt                      P=0.0559 (776 docs)
    # rec.motorcycles                P=0.0545 (757 docs)
    # soc.religion.christian         P=0.0540 (750 docs)
    # sci.space                      P=0.0536 (745 docs)
    # sci.med                        P=0.0536 (745 docs)
    # rec.sport.hockey               P=0.0534 (742 docs)
    # comp.windows.x                 P=0.0524 (728 docs)
    # rec.sport.baseball             P=0.0516 (716 docs)
    # sci.electronics                P=0.0510 (708 docs)
    # rec.autos                      P=0.0508 (705 docs)
    # ... and 10 more categories

    # 2. Computing word likelihoods using MapReduce...
    # Vocabulary size: 79,836 unique words
    # Total words: 1,913,537

    # ============================================================
    # TESTING MODEL
    # ============================================================

    # ============================================================
    # RESULTS
    # ============================================================
    # Test samples: 3576
    # Correct: 2824
    # Accuracy: 0.7897 (78.97%)

    # ============================================================
    # TOP 10 CLASSES BY ACCURACY
    # ============================================================
    # rec.sport.hockey               94.25%  (164/174)
    # soc.religion.christian         93.45%  (157/168)
    # talk.politics.guns             91.62%  (164/179)
    # talk.politics.mideast          91.57%  (163/178)
    # rec.sport.baseball             91.26%  (167/183)
    # sci.crypt                      89.47%  (170/190)
    # talk.politics.misc             88.49%  (123/139)
    # rec.motorcycles                85.79%  (169/197)
    # sci.space                      85.26%  (162/190)
    # comp.windows.x                 85.07%  (171/201)

    # ============================================================

# Report: https://github.com/Kushal3121/CS6350-Assignments/blob/main/Assignment%202/kxc240000/2.%20Naive%20Bayes%20Classifier/report.md