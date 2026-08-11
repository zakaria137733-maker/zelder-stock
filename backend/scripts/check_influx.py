import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from services.influx import get_influx_client

client = get_influx_client()
query_api = client.query_api()

flux = f"""
from(bucket: "{settings.influx_bucket}")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "prices")
  |> limit(n: 5)
"""
tables = query_api.query(flux)
found = False
for t in tables:
    for r in t.records:
        print("Price record:", r.values.get("ticker"), r.values.get("_field"), r.values.get("_value"), r.get_time())
        found = True
if not found:
    print("No price data found in last 1h")
client.close()
