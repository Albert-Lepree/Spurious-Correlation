import databricks_ingest
import json
import pandas as pd

# Load queries
with open("databricks_seed.json", "r") as f:
    queries = json.load(f)

# Load Kaggle CSV locally (download from Databricks first)
kaggle_df = pd.read_csv("spurious_news.csv")
print(f"  Rows: {len(kaggle_df)}")

# Connect to Databricks for the rest
conn = databricks_ingest.init_databricks()

google_df = databricks_ingest.query_databricks(conn, queries["google_news"])
google_df.to_csv("data/google_news.csv", index=False)
print(f"  Rows: {len(google_df)}")

# print("Pulling spx_data...")
# spx_df = query_databricks(conn, queries["spx_data"])
# spx_df.to_csv("data/spx_data.csv", index=False)
# print(f"  Rows: {len(spx_df)}")

# print("Pulling ndx_data...")
# ndx_df = query_databricks(conn, queries["ndx_data"])
# ndx_df.to_csv("data/ndx_data.csv", index=False)
# print(f"  Rows: {len(ndx_df)}")

# print("Pulling vix_data...")
# vix_df = query_databricks(conn, queries["vix_data"])
# vix_df.to_csv("data/vix_data.csv", index=False)
# print(f"  Rows: {len(vix_df)}")

# Close
conn.close()