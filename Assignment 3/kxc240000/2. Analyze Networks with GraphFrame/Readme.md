## How to Run

Run the script using the following command:

```bash
spark-submit \
 --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
 --driver-memory 4g \
 --executor-memory 4g \
 analyze_networks.py
```
