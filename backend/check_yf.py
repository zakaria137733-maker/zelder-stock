import yfinance as yf

t = yf.Ticker("AAPL")
hist = t.history(period="1d", interval="1h")
print("Columns:", hist.columns.tolist())
print("Index type:", type(hist.index[0]))
print("First 3 timestamps:")
for ts in hist.index[:3]:
    print(" ", ts, "| tzinfo:", ts.tzinfo)
print("Last timestamp:", hist.index[-1])