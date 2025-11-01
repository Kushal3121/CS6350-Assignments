import os, json
from pyspark.sql import SparkSession, functions as F, types as T

# --- Lazy spaCy loader (loads once per executor) ---
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# --- UDF to extract named entities ---
@F.udf(returnType=T.ArrayType(T.StringType()))
def extract_entities(text):
    if text is None:
        return []
    nlp = get_nlp()
    doc = nlp(text)
    labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "FAC"}
    ents = [ent.text.strip() for ent in doc.ents if ent.label_ in labels and ent.text.strip()]
    return ents


def main():
    bootstrap = "localhost:9092"
    in_topic  = "topic1"
    out_topic = "topic2"

    spark = (
        SparkSession.builder
        .appName("NER_Stream_Processor")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Read from Kafka (real-time text source)
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", in_topic)
        # Use 'latest' so we don’t keep re-reading old messages
        .option("startingOffsets", "latest")
        .load()
    )

    # 2. Parse JSON messages
    schema = T.StructType([
        T.StructField("id", T.StringType()),
        T.StructField("author", T.StringType()),
        T.StructField("created_utc", T.DoubleType()),
        T.StructField("text", T.StringType()),
        T.StructField("subreddit", T.StringType()),
    ])
    parsed = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("j")).select("j.*")

    # 3. Extract named entities using spaCy
    with_ents = parsed.withColumn("entities", extract_entities(F.col("text")))

    # 4. Explode entities and count frequencies
    exploded = with_ents.select(F.explode(F.col("entities")).alias("entity"))
    counts = exploded.groupBy("entity").count().withColumnRenamed("count", "running_count")

    # 5. Prepare output to Kafka topic2 as JSON
    out = counts.select(
        F.to_json(
            F.struct(
                F.col("entity"),
                F.col("running_count").alias("count"),
                F.current_timestamp().alias("ts")
            )
        ).alias("value")
    )

    # 6. Write continuous stream to Kafka
    query = (
        out.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("topic", out_topic)
        .outputMode("update")                # send only updates, not full table
        .trigger(processingTime="30 seconds")# refresh every 30s
        .option("checkpointLocation", "./chk/ner_ckpt")
        .start()
    )

    # Print entities to console for debugging
    debug_query = (
        counts.writeStream
        .outputMode("update")
        .format("console")
        .trigger(processingTime="30 seconds")
        .start()
    )

    # Keep running
    query.awaitTermination()
    debug_query.awaitTermination()


if __name__ == "__main__":
    main()
