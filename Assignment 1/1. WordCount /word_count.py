# ------------------------------------------------------------------ Big Data Assignment 1 ----------------------------------------------------------------- #
# Name - Kushal Choudhary
# Net ID - kxc240000
# ---------------------------------------------------------- Named Entity Recognition Word Count ----------------------------------------------------------- #

from pyspark import SparkContext
import os
import spacy

# Download the text file from Project Gutenberg
# Pride and Prejudice
BOOK_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
if not os.path.exists("book.txt"):
    os.system(f"wget {BOOK_URL} -O book.txt")

# Initialize Spark
sc = SparkContext(appName="NamedEntityWordCount")
sc.setLogLevel("ERROR")

# Load file into RDD
text_rdd = sc.textFile("book.txt")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to extract named entities, filter by type
def extract_entities(line):
    doc = nlp(line)
    return [ent.text.strip() for ent in doc.ents if ent.label_ in {"PERSON", "GPE", "ORG", "LOC"}]

# Extract entities
entities_rdd = text_rdd.flatMap(extract_entities)

# MapReduce word count
entity_pairs = entities_rdd.map(lambda ent: (ent, 1))
entity_counts = entity_pairs.reduceByKey(lambda a, b: a + b)

# Sort in descending order of frequency
sorted_entities = entity_counts.sortBy(lambda x: x[1], ascending=False)

# Collect and print top 20 entities
top_entities = sorted_entities.take(20)
print("\n=== Top 20 Named Entities (Filtered) ===")
for entity, count in top_entities:
    print(f"{entity}: {count}")

sc.stop()


# Output:
    # === Top 20 Named Entities (Filtered) ===
    # Elizabeth: 633
    # Darcy: 364
    # Jane: 294
    # Bingley: 243
    # Bennet: 232
    # Wickham: 190
    # Collins: 189
    # Lydia: 163
    # Lizzy: 95
    # Gardiner: 94
    # Lady Catherine: 86
    # Charlotte: 77
    # Netherfield: 71
    # Kitty: 56
    # Longbourn: 55
    # London: 55
    # Meryton: 51
    # Rosings: 45
    # Miss Bingley: 41
    # Pemberley: 39

