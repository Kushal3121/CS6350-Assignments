# Real-Time Named Entity Recognition using Spark Streaming, Kafka, and Elastic Stack

## Overview

This project implements a **real-time data processing pipeline** that collects live text data using the **NewsAPI**, extracts **Named Entities (NER)** using **SpaCy** and **PySpark Structured Streaming**, and visualizes entity trends using the **Elastic Stack (Logstash, Elasticsearch, and Kibana)**.

The purpose of this system is to demonstrate end-to-end real-time analytics using distributed big data technologies.

---

## Architecture

**Data Flow:**  
`NewsAPI → Kafka (topic1) → Spark Structured Streaming → Kafka (topic2) → Logstash → Elasticsearch → Kibana`

### Components:

- **Producer:** Fetches live news headlines from NewsAPI and sends them to Kafka (`topic1`).
- **Spark Streaming:** Reads data from Kafka (`topic1`), performs NER using SpaCy, counts entity occurrences, and publishes results to Kafka (`topic2`).
- **Logstash:** Reads processed entity data from Kafka (`topic2`) and indexes it into Elasticsearch.
- **Kibana:** Visualizes the entity counts and their trends over time.

---

## Requirements

### Prerequisites

Ensure the following are installed and running:

- **Python 3.9+**
- **Apache Kafka & Zookeeper**
- **Apache Spark 3.5+**
- **Elasticsearch 7.17+**
- **Kibana 7.17+**
- **Logstash 9.x**

### Python Libraries

Install the required Python dependencies:

```bash
pip install pyspark spacy confluent-kafka requests
python -m spacy download en_core_web_sm
```

## Steps to execute the assignment

### 1. Start Zookeeper

```bash
zookeeper-server-start config/zookeeper.properties
```

### 2. Start Kafka

```bash
kafka-server-start config/server.properties
```

### 3. Create Kafka Topics

```bash
kafka-topics --create --topic topic1 --bootstrap-server localhost:9092
kafka-topics --create --topic topic2 --bootstrap-server localhost:9092
```

### Verify topics

```bash
kafka-topics --list --bootstrap-server localhost:9092
```

### 4. Start Elasticsearch

```bash
elasticsearch-full
```

### 5. Start Kibana

```bash
/opt/homebrew/opt/kibana-full/bin/kibana
```

### 6. Start the NewsAPI Producer

```bash
python newsapi_producer.py
```

## 7. Start the Spark Structured Streaming Job

```bash
spark-submit \
 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
 stream_ner_to_counts.py
```

### 8. Start Logstash (Kafka → Elasticsearch)

```bash
logstash -f logstash-kafka.conf
```
