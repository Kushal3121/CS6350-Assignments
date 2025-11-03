# ---------------------------------------------------------------
# CS 6350 - Assignment 3 - Graph Analysis using GraphFrames
# Name  : Kushal Choudhary
# NetID : kxc240000
# ---------------------------------------------------------------

import os
import subprocess
from pyspark.sql import SparkSession, functions as F
from graphframes import GraphFrame

# ---------------------------------------------------------------
# Download Dataset (no hardcoded paths)
# ---------------------------------------------------------------

DATA_URL = "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"
DATA_DIR = "data"
RAW_GZ_PATH = os.path.join(DATA_DIR, "soc-Epinions1.txt.gz")
RAW_TXT_PATH = os.path.join(DATA_DIR, "soc-Epinions1.txt")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(RAW_GZ_PATH) and not os.path.exists(RAW_TXT_PATH):
    subprocess.run(["wget", DATA_URL, "-O", RAW_GZ_PATH], check=True)

if os.path.exists(RAW_GZ_PATH) and not os.path.exists(RAW_TXT_PATH):
    subprocess.run(["gunzip", "-f", RAW_GZ_PATH], check=True)

# ---------------------------------------------------------------
# Initialize Spark + GraphFrames
# ---------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("GraphAnalysis-KushalChoudhary")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# ---------------------------------------------------------------
# Load and Parse Data (handles comments + variable spaces)
# ---------------------------------------------------------------

edges_df = (
    spark.read.text(RAW_TXT_PATH)
    .filter(~F.col("value").startswith("#"))            
    .filter(F.length(F.col("value")) > 0)               
    .select(
        F.split(F.col("value"), r"\s+")[0].alias("src"),
        F.split(F.col("value"), r"\s+")[1].alias("dst")
    )
    .dropna()
)

# Create distinct vertex DataFrame
vertices_src = edges_df.select(F.col("src").alias("id"))
vertices_dst = edges_df.select(F.col("dst").alias("id"))
vertices_df = vertices_src.union(vertices_dst).distinct()

print("Loaded edges:", edges_df.count())
print("Loaded vertices:", vertices_df.count())

# ---------------------------------------------------------------
# Create GraphFrame
# ---------------------------------------------------------------

g = GraphFrame(vertices_df, edges_df)
print("\nGraphFrame created successfully!")

# Create output folder
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------
# Run Graph Algorithms
# ---------------------------------------------------------------

# (a) Top 5 nodes by outdegree
outdeg = g.outDegrees.orderBy(F.desc("outDegree")).limit(5)
outdeg.show()
outdeg.write.mode("overwrite").csv("output/top5_outdegree")

# (b) Top 5 nodes by indegree
indeg = g.inDegrees.orderBy(F.desc("inDegree")).limit(5)
indeg.show()
indeg.write.mode("overwrite").csv("output/top5_indegree")

# (c) PageRank (top 5)
pagerank = g.pageRank(resetProbability=0.15, maxIter=10)
pr_top = pagerank.vertices.orderBy(F.desc("pagerank")).limit(5)
pr_top.show()
pr_top.write.mode("overwrite").csv("output/top5_pagerank")

# (d) Connected Components (top 5 largest)
os.makedirs("checkpoint", exist_ok=True)
spark.sparkContext.setCheckpointDir("checkpoint")

cc = g.connectedComponents()
cc_count = cc.groupBy("component").count().orderBy(F.desc("count")).limit(5)
cc_count.show()
cc_count.write.mode("overwrite").csv("output/top5_components")

# (e) Triangle Count (top 5)
triangles = g.triangleCount()
tri_top = triangles.orderBy(F.desc("count")).limit(5)
tri_top.show()
tri_top.write.mode("overwrite").csv("output/top5_trianglecount")


print("Output files:")
for f in os.listdir("output"):
    print("  -", f)

# ---------------------------------------------------------------
# Write Summary Results File
# ---------------------------------------------------------------
summary_path = "output/summary_results.txt"
with open(summary_path, "w") as f:
    f.write("=== CS 6350 Assignment 3: Graph Analysis ===\n\n")

    f.write("Top 5 Nodes by OutDegree:\n")
    f.write(str(outdeg.toPandas()) + "\n\n")

    f.write("Top 5 Nodes by InDegree:\n")
    f.write(str(indeg.toPandas()) + "\n\n")

    f.write("Top 5 Nodes by PageRank:\n")
    f.write(str(pr_top.toPandas()) + "\n\n")

    f.write("Top 5 Components (Largest):\n")
    f.write(str(cc_count.toPandas()) + "\n\n")

    f.write("Top 5 Vertices by Triangle Count:\n")
    f.write(str(tri_top.toPandas()) + "\n\n")

print(f"Summary file created: {summary_path}")

spark.stop()
