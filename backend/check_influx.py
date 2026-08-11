from influxdb_client import InfluxDBClient
from config import settings

client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
query_api = client.query_api()

flux = """
from(bucket: "sentiment_scores")
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