#!/usr/bin/env python3
import json, time, requests
from confluent_kafka import Producer

# ---------- CONFIG ----------
API_KEY   = "abfc3bfb8a7549c8b52c6d355eebd7a2"   # <-- your key
BOOTSTRAP = "localhost:9092"
TOPIC     = "topic1"
LANGUAGE  = "en"
COUNTRY   = "us"   # change to "in", "gb", etc. if you want

# ---------- SETUP ----------
producer = Producer({"bootstrap.servers": BOOTSTRAP})
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country={COUNTRY}&language={LANGUAGE}&apiKey={API_KEY}"

def delivery_report(err, msg):
    if err:
        print(f"❌ Delivery failed: {err}")
    else:
        print(f"✅ Sent to {msg.topic()} [p{msg.partition()}] @ offset {msg.offset()}")

print(f"🚀 Streaming live headlines from NewsAPI ({COUNTRY}) → Kafka:{TOPIC}")
print("   (Press Ctrl+C to stop)\n")

# ---------- MAIN LOOP ----------
try:
    while True:
        try:
            res = requests.get(NEWS_URL)
            data = res.json()
            articles = data.get("articles", [])

            for art in articles:
                title = art.get("title") or ""
                desc  = art.get("description") or ""
                text  = (title + ". " + desc).strip()

                if not text or len(text) < 10:
                    continue

                payload = {
                    "id": art.get("url"),
                    "author": art.get("source", {}).get("name"),
                    "created_utc": time.time(),
                    "text": text,
                    "subreddit": "newsapi",  # just a field label
                }

                producer.produce(TOPIC, json.dumps(payload).encode("utf-8"), callback=delivery_report)
                producer.poll(0)

            producer.flush()
            print(f"📰 Sent {len(articles)} new headlines at {time.strftime('%H:%M:%S')}\n")
            time.sleep(30)  # fetch new headlines every 30 sec

        except Exception as e:
            print("⚠️ Error fetching/sending batch:", e)
            time.sleep(15)

except KeyboardInterrupt:
    print("\n⏹️ Stopping stream...")
finally:
    producer.flush()
    print("✅ All pending messages delivered.")
