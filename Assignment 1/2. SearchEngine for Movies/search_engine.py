from pyspark import SparkContext
import os, math, re, glob
import nltk
from collections import defaultdict

# Download stopwords if not already
nltk.download("stopwords")
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

# Step 1: Download & Extract dataset
DATA_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"

if not os.path.exists("MovieSummaries.tar.gz"):
    print("Downloading dataset...")
    os.system(f"wget -q {DATA_URL} -O MovieSummaries.tar.gz")

print("Extracting dataset...")
os.system("tar -xzf MovieSummaries.tar.gz")

# Helper: find file no matter if it's in root or MovieSummaries/
def find_file(fname):
    matches = glob.glob(f"**/{fname}", recursive=True)
    if matches:
        return matches[0]
    else:
        raise FileNotFoundError(f"{fname} not found after extraction")

plot_file = find_file("plot_summaries.txt")
meta_file = find_file("movie.metadata.tsv")

# Step 2: Initialize Spark
sc = SparkContext(appName="MoviePlotSearchEngine")
sc.setLogLevel("ERROR")

# Step 3: Load Data
# Format of plot_summaries.txt → (movieID, plot summary)
plots = sc.textFile(plot_file) \
          .map(lambda line: line.split("\t", 1)) \
          .filter(lambda x: len(x) == 2)

# Metadata (movieID → title)
metadata = sc.textFile(meta_file) \
             .map(lambda line: line.split("\t")) \
             .map(lambda x: (x[0], x[2]))
movie_titles = metadata.collectAsMap()

# Step 4: Tokenization & Stopword Removal
def tokenize(text):
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())  # keep only words
    return [t for t in tokens if t not in STOPWORDS]

tokenized = plots.mapValues(tokenize)

# Step 5: Compute Term Frequency (TF)
tf = tokenized.flatMap(lambda x: [((x[0], term), 1) for term in x[1]]) \
              .reduceByKey(lambda a, b: a + b)

# Step 6: Compute Document Frequency (DF)
term_docs = tf.map(lambda x: (x[0][1], x[0][0])) \
              .distinct() \
              .map(lambda x: (x[0], 1)) \
              .reduceByKey(lambda a, b: a + b)

N = plots.count()
idf = term_docs.mapValues(lambda df: math.log(N / df))

# Step 7: Compute TF-IDF
tfidf = tf.map(lambda x: (x[0][1], (x[0][0], x[1]))) \
          .join(idf) \
          .map(lambda x: ((x[1][0][0], x[0]), x[1][0][1] * x[1][1]))
# Format → ((docID, term), tfidf)

# Build doc → {term: tfidf} dictionary
doc_vectors = tfidf.map(lambda x: (x[0][0], (x[0][1], x[1]))) \
                   .groupByKey() \
                   .mapValues(dict)
doc_vectors.cache()

# Step 8: Cosine Similarity
def cosine_similarity(query_vec, doc_vec):
    dot = sum(query_vec[t] * doc_vec.get(t, 0) for t in query_vec)
    q_norm = math.sqrt(sum(v*v for v in query_vec.values()))
    d_norm = math.sqrt(sum(v*v for v in doc_vec.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)

# Step 9: Queries
with open("queries.txt") as f:
    queries = [line.strip() for line in f if line.strip()]

for q in queries:
    print("\n=== Query:", q, "===")
    terms = tokenize(q)

    if len(terms) == 1:  # Single-term query → rank by tf-idf
        term = terms[0]
        results = tfidf.filter(lambda x: x[0][1] == term) \
                       .map(lambda x: (x[0][0], x[1])) \
                       .top(10, key=lambda x: x[1])
        for docID, score in results:
            print(f"{movie_titles.get(docID, 'Unknown')}: {score:.4f}")

    else:  # Multi-term query → cosine similarity
        tf_counts = defaultdict(int)
        for t in terms:
            tf_counts[t] += 1

        query_vec = {}
        for t, count in tf_counts.items():
            idf_val = idf.lookup(t)
            if idf_val:
                query_vec[t] = count * idf_val[0]

        sims = doc_vectors.map(lambda x: (x[0], cosine_similarity(query_vec, x[1])))
        top_docs = sims.top(10, key=lambda x: x[1])

        for docID, sim in top_docs:
            if sim > 0:
                print(f"{movie_titles.get(docID, 'Unknown')}: {sim:.4f}")

sc.stop()

# Output:

    # === Query: vampire ===
    # Les Vampires: 129.3742
    # The Breed: 109.4705
    # The Return of the Vampire: 89.5667
    # Blade: 79.6149
    # Blade: Trinity: 74.6390
    # Modern Vampires: 64.6871
    # The Lost Boys: 64.6871
    # Lost Boys: The Thirst: 59.7112
    # Vampires: The Turning: 54.7352
    # Demon Under Glass: 54.7352

    # === Query: romantic ===
    # Genova: 13.9550
    # The Little Rascals: 13.9550
    # Rome Adventure: 13.9550
    # The Heart Desires: 10.4662
    # Monkey Shines: 10.4662
    # Once: 10.4662
    # Gidget: 10.4662
    # Cardcaptor: The Movie 2: 10.4662
    # He's Just Not That Into You: 10.4662
    # A Cold Wind in August: 10.4662

    # === Query: comedy ===
    # Talkin' Dirty After Dark: 17.2493
    # Man on the Moon: 17.2493
    # General Motors 50th Anniversary Show: 17.2493
    # Where the Truth Lies: 17.2493
    # Hollywood Outlaw Movie: 17.2493
    # Punchline: 12.9370
    # Thousands Cheer: 12.9370
    # Cinta Kura Kura: 12.9370
    # Mahaul Theek Hai: 12.9370
    # When Stand Up Stood Out: 12.9370

    # === Query: horror ===
    # Garo: Red Requiem: 44.7620
    # The Last Horror Film: 20.3464
    # The Pagemaster: 20.3464
    # Raat: 16.2771
    # The Pit and the Pendulum: 16.2771
    # Fright Night: 16.2771
    # Kiba Gaiden: 16.2771
    # Microwave Massacre: 16.2771
    # Tenebrae: 12.2078
    # Free Jimmy: 12.2078

    # === Query: war ===
    # Breaker Morant: 28.4972
    # North and South: 26.1224
    # Born on the Fourth of July: 26.1224
    # Hiroshima: 23.7476
    # The Life and Death of Colonel Blimp: 23.7476
    # Oh! What a Lovely War: 23.7476
    # InuYasha the Movie: Fire on the Mystic Island: 23.7476
    # Birthday Boy: 21.3729
    # The War You Don't See: 21.3729
    # Week-End at the Waldorf: 21.3729

    # === Query: funny movie with action scenes ===
    # The Major Lied 'Til Dawn: 0.2136
    # Kottarathil Kuttibhootham: 0.2039
    # The Daredevil Men: 0.1964
    # Lu and Bun: 0.1963
    # Funny Man: 0.1948
    # Action Man: Robot Atak: 0.1922
    # Aanaval mothiram: 0.1710
    # Unknown: 0.1524
    # If You Only Knew: 0.1508
    # Ullathai Allitha: 0.1501

    # === Query: romantic comedy drama ===
    # Blind Love: 0.3593
    # Mango Soufflé: 0.3422
    # Marriage Story: 0.2981
    # Eat me!: 0.2979
    # Loop: 0.2976
    # I Was a Teenage Zombie: 0.2923
    # Bhakta Prahlada: 0.2558
    # Chatham: 0.2533
    # Talkin' Dirty After Dark: 0.2496
    # Mister Ten Per Cent: 0.2386

    # === Query: science fiction future ===
    # Storytelling: 0.4042
    # Orders Are Orders: 0.3696
    # Outside: 0.3442
    # PROXIMA: 0.3421
    # Time Adventure: Zeccho 5-byo Mae: 0.3383
    # Project Shadowchaser III: 0.3336
    # Remote Control: 0.3323
    # Mother: 0.2901
    # Polygon: 0.2658
    # Should I Really Do It?: 0.2539

    # === Query: family friendship love ===
    # Yes or No: 0.2599
    # Yaar Annmulle Movie: 0.2592
    # Kini and Adams: 0.2442
    # Birthday: 0.2431
    # Ceremony: 0.2110
    # Submarine: 0.2034
    # Accidental Friendship: 0.1997
    # Unknown: 0.1963
    # July 14th: 0.1939
    # Kohi Mero: 0.1916

    # === Query: murder mystery detective ===
    # Accused: 0.3891
    # My Favorite Brunette: 0.3143
    # True Crime: 0.2955
    # Follow Me!: 0.2842
    # Hotel Chelsea: 0.2790
    # Three Steps in the Dark: 0.2706
    # Stripped to Kill: 0.2664
    # Sahasram: 0.2516
    # Inflatable Sex Doll of the Wastelands: 0.2467
    # The Comeback: 0.2334