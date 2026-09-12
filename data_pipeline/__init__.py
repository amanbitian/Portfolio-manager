"""Raw-data ingestion pipeline for the ML portfolio allocator.

Fetches market prices, corporate actions, fundamentals, macro series and news for
the current Nifty 1000 universe and writes them as partitioned Parquet under the
directory pointed to by ``STOCK_DATA_ROOT`` (default ``F:/quants project/stock Data``).

Run ``py -3 -m data_pipeline.run --help`` for the CLI.
"""

from .config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
