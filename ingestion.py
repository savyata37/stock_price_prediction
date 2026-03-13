

import pandas as pd
import logging

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_csv_chunked(file_path, chunk_size=10000):
    """
    Load large CSV files using chunking
    """
    chunks = []
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logging.info(f"CSV loaded successfully with chunking (shape: {df.shape})")
        return df
    except Exception as e:
        logging.error("CSV loading failed: %s", e)
        raise


def load_data(file_path="sp500_data.csv", chunk_size=None):
    """
    Unified data loader.
    - If chunk_size is None → normal fast read
    - If chunk_size is set → uses chunking
    """
    if chunk_size is None:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Normal CSV load successful (shape: {df.shape})")
            return df
        except Exception as e:
            logging.error("Normal CSV load failed: %s", e)
            raise
    else:
        return load_csv_chunked(file_path, chunk_size)


