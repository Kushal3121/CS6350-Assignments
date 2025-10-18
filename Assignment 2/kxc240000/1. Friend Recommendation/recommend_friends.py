# ------------------------------------------------------------------ Big Data Assignment 2 ----------------------------------------------------------------- #
# Name - Kushal Choudhary
# Net ID - kxc240000
# ------------------------------------------------------- Friend Recommendation using Mutual Friends ------------------------------------------------------- #


import argparse
import os
import urllib.request
from pyspark.sql import SparkSession
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="URL or local file path")
    p.add_argument("--sample", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def normalize_pair(a, b):
    return (a, b) if a < b else (b, a)

def download_if_url(input_path):
    """Download file if it's a URL, return local path"""
    if input_path.startswith('http'):
        local_file = "livejournal_data.txt"
        if not os.path.exists(local_file):
            print(f"Downloading {input_path}...")
            urllib.request.urlretrieve(input_path, local_file)
        else:
            print(f"Using cached file: {local_file}")
        return local_file
    return input_path

if __name__ == "__main__":
    args = parse_args()
    
    # Download file if URL
    local_input = download_if_url(args.input)
    
    print("\nStarting Spark...")
    spark = (SparkSession.builder
             .appName("FriendRecommendation")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    print("Loading data from local file...")
    lines = sc.textFile(local_input)
    
    def parse_line(line):
        parts = line.strip().split('\t')
        if len(parts) < 2 or not parts[0].strip():
            return None
        user = parts[0].strip()
        friends = [f.strip() for f in parts[1].split(',') if f.strip()]
        return (user, friends) if friends else None
    
    user_friends = lines.map(parse_line).filter(lambda x: x is not None).cache()
    
    print("Sampling users...")
    
    import random
    random.seed(args.seed)
    
    # Get first 1000 users to sample from
    first_batch = user_friends.zipWithIndex().filter(lambda x: x[1] < 1000).map(lambda x: x[0][0]).collect()
    sampled_users = random.sample(first_batch, min(args.sample, len(first_batch)))
    
    print(f"Selected {len(sampled_users)} users: {sampled_users}")
    
    sampled_bc = sc.broadcast(set(sampled_users))
    
    print("Computing existing friendships...")
    
    existing_edges = (user_friends
                      .filter(lambda x: x[0] in sampled_bc.value)
                      .flatMap(lambda x: [normalize_pair(x[0], f) for f in x[1]])
                      .distinct()
                      .map(lambda p: (p, True)))
    
    print("Finding mutual friends...")
    
    def generate_pairs(x):
        user, friends = x
        pairs = []
        for i in range(len(friends)):
            for j in range(i + 1, len(friends)):
                if friends[i] != friends[j]:
                    pairs.append((normalize_pair(friends[i], friends[j]), 1))
        return pairs
    
    mutual_counts = (user_friends
                     .flatMap(generate_pairs)
                     .reduceByKey(lambda a, b: a + b))
    
    print("Building recommendations...")
    
    # Remove existing friendships
    non_friends = (mutual_counts
                   .leftOuterJoin(existing_edges)
                   .filter(lambda x: x[1][1] is None)
                   .map(lambda x: (x[0], x[1][0])))
    
    # Expand to user-centric
    def expand_pairs(x):
        pair, count = x
        results = []
        if pair[0] in sampled_bc.value:
            results.append((pair[0], (pair[1], count)))
        if pair[1] in sampled_bc.value:
            results.append((pair[1], (pair[0], count)))
        return results
    
    recommendations = (non_friends
                       .flatMap(expand_pairs)
                       .groupByKey()
                       .mapValues(lambda recs: sorted(list(recs), 
                                  key=lambda x: (-x[1], int(x[0]) if x[0].isdigit() else x[0]))[:10]))
    
    results = recommendations.collect()
    
    print("FRIEND RECOMMENDATIONS")
    
    for user, recs in sorted(results, key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        rec_list = ','.join([r[0] for r in recs])
        print(f"{user}\t{rec_list}")
    
    print(f"\n{len(results)} users processed successfully!")
    print("="*60)
    
    spark.stop()

# Output:
    # FRIEND RECOMMENDATIONS

    # 25      1,2,3,4,5,6,7,8,9,10
    # 2203    2169,2199,0,21,98,543,575,1450,2119,2120
    # 2572    29521,1421,6653,4431,6654,13,123,1790,4400,4527
    # 16691   8083,8091,8105,11896,8076,11889,11890,11895,8075,8077
    # 18071   13834,19051,0,17,223,242,368,1489,2130,2256
    # 29581   16,0,12,30,6027,9822,12260,13081,13109,13793
    # 29714   29694,29689,29700,29707,29711,29697,29704,29709,29712,34892
    # 30692   19,439,667,1085,1113,8685,11562,15712,16532,19150
    # 32278   30305,32004,32009,32008,32070,49895,3578,12221,12578,14029
    # 42697   27383,31989,42699,42703,1857,19009,42693,42695,42702,42706

    # 10 users processed successfully!