from numerapi import NumerAPI
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from tqdm import tqdm

napi = NumerAPI()

parquet_file = pq.ParquetFile("v4.3/train_int8.parquet")


model = lgb.LGBMRegressor(
      n_estimators=2000,
      learning_rate=0.01,
      max_depth=5,
      num_leaves=2 ** 5,
      colsample_bytree=0.1
)



batch_size = 100000
def read_parquet_in_chunks(parquet_file, batch_size):
    i = 0
    while True:
        try:
            # Read a batch
            table = parquet_file.read_row_group(i)
            yield table.to_pandas()
            i += 1
        except IndexError:
            # Stop iteration if there are no more row groups
            break
first_pass = True
# Train the model incrementally on each chunk
for chunk in tqdm(read_parquet_in_chunks(parquet_file, batch_size),total=parquet_file.num_row_groups, desc="Processing row groups"):
    
    features = [f for f in chunk.columns if "feature" in f]
    if "target" in chunk.columns:  # Ensure the target column is present
        if first_pass:
            # Fit the model for the first time without init_model
            model.fit(chunk[features], chunk["target"])
            first_pass = False
        else:  # Ensure the target column is present
            model.fit(
                chunk[features],  # Feature columns
                chunk["target"],  # Target column
                init_model=model if model.booster_ is not None else None,  # Incremental training
                keep_training_booster=True
            )

# Optionally, save the model or perform further operations
model.booster_.save_model("trained_model.txt")