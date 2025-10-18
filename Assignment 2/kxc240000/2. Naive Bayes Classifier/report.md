# Naive Bayes Text Classification Algorithm

**Author:** Kushal Choudhary  
**Net ID:** kxc240000  
**Dataset:** 20 Newsgroups (http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)

---

## Algorithm Overview

This algorithm implements a Naive Bayes text classifier using MapReduce with Spark RDDs. It classifies documents into categories based on word occurrence probabilities, using Bayes' theorem with the naive independence assumption.

**Formula:** `P(Class|Document) ∝ P(Class) × ∏ P(word|Class)`

---

## Algorithm Steps

1. **Preprocess Text**

   - Tokenize documents into words
   - Remove stop words, punctuation, and short tokens
   - Convert to lowercase

2. **Split Train/Test**

   - Randomly split data (80% train, 20% test)
   - Cache both datasets for reuse

3. **Calculate Class Priors P(Class)**

   - Count documents per class
   - Compute probability: P(Class) = count(Class) / total_documents

4. **Calculate Word Likelihoods P(word|Class)**

   - Count word occurrences in each class using MapReduce
   - Apply Laplace smoothing to handle unseen words
   - Formula: P(word|Class) = (count(word,Class) + 1) / (total_words_Class + vocab_size)

5. **Make Predictions**

   - For each test document, compute score for each class
   - Use log probabilities: score = log P(Class) + Σ log P(word|Class)
   - Predict class with highest score

6. **Evaluate Results**
   - Compare predictions with actual labels
   - Calculate overall accuracy and per-class metrics

---

## Pseudo-Code

```
FUNCTION naive_bayes_classifier(documents, test_split):

    // Step 1: Preprocess
    processed = documents
                .MAP(tokenize_and_clean)
                .FILTER(valid documents)

    train, test = processed.RANDOMSPLIT([0.8, 0.2])

    // Step 2: Calculate class priors
    class_counts = train
                   .MAP(doc -> (doc.label, 1))
                   .REDUCEBYKEY(sum)

    priors = {class: count/total for class, count in class_counts}

    // Step 3: Calculate word likelihoods (MapReduce)
    word_counts = train
                  .FLATMAP(doc -> [((doc.label, word), 1) for word in doc.tokens])
                  .REDUCEBYKEY(sum)

    class_word_totals = train
                        .FLATMAP(doc -> [(doc.label, len(doc.tokens))])
                        .REDUCEBYKEY(sum)

    vocabulary_size = train.FLATMAP(doc -> doc.tokens).DISTINCT().COUNT()

    // Step 4: Prediction function
    FUNCTION predict(tokens):
        scores = {}
        FOR each class:
            score = LOG(priors[class])
            FOR each word in tokens:
                word_count = word_counts.GET((class, word), 0)
                likelihood = (word_count + 1) / (class_word_totals[class] + vocabulary_size)
                score += LOG(likelihood)
            scores[class] = score
        RETURN argmax(scores)

    // Step 5: Evaluate
    predictions = test.MAP(doc -> (doc.label, predict(doc.tokens)))

    accuracy = predictions.FILTER(true == predicted).COUNT() / test.COUNT()

    RETURN accuracy, predictions

// Helper function
FUNCTION tokenize_and_clean(text):
    text = LOWERCASE(text)
    text = REMOVE_PUNCTUATION(text)
    tokens = SPLIT(text)
    tokens = REMOVE_STOPWORDS(tokens)
    RETURN FILTER(tokens, length > 2)
```

---

## Results Summary

### Dataset Statistics

- **Total Documents:** 18,846
- **Training Set:** 15,076 documents (80%)
- **Test Set:** 3,770 documents (20%)
- **Number of Classes:** 20 categories
- **Vocabulary Size:** 79,836 unique words

### Model Performance

- **Overall Accuracy:** 78.97%
- **Baseline (Random Guess):** 5.00%

### Top Performing Classes

| Category               | Accuracy |
| ---------------------- | -------- |
| rec.sport.hockey       | 94.25%   |
| soc.religion.christian | 93.45%   |
| talk.politics.guns     | 91.62%   |
| talk.politics.mideast  | 91.57%   |
| rec.sport.baseball     | 91.26%   |

### Class Priors (Top 5)

- comp.graphics: 5.24%
- rec.sport.baseball: 5.16%
- sci.electronics: 5.10%
- rec.autos: 5.08%
- comp.windows.x: 5.03%

---

## Key MapReduce Operations

1. **Map:** Tokenize documents, emit (label, 1) for priors
2. **FlatMap:** Emit ((class, word), 1) for each word occurrence
3. **ReduceByKey:** Aggregate word counts per class
4. **Filter:** Apply predictions and count correct classifications

---

## Complexity Analysis

- **Training Time:** O(N × W) where N = documents, W = avg words per document
- **Prediction Time:** O(C × W) where C = number of classes
- **Space:** O(V × C) where V = vocabulary size
- **MapReduce Stages:** 3 main shuffles (reduceByKey for counts)
