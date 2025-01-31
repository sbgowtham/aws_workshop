import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq

# Define file paths
orc_file_path = "large_sample.orc"
parquet_file_path = "large_sample.parquet"
csv_file_path = "large_sample.csv"

# Generate large dataset (~100 MB+)
num_rows = 5_000_000  # 5 million rows
data = {
    "id": np.arange(1, num_rows + 1),
    "name": np.random.choice(["Alice", "Bob", "Charlie", "David", "Eve"], num_rows),
    "value": np.random.uniform(10.0, 1000.0, num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert DataFrame to PyArrow Table
table = pa.Table.from_pandas(df)

# Save as ORC file
with pa.OSFile(orc_file_path, "wb") as f:
    orc.write_table(table, f)

# Save as Parquet file
pq.write_table(table, parquet_file_path)

# Save as CSV file
df.to_csv(csv_file_path, index=False)

print(f"Files created:\n{orc_file_path}\n{parquet_file_path}\n{csv_file_path}")
