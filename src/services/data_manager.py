# services/data/data_manager.py
from typing import Dict, Any, List, Optional
import pandas as pd
import logging
from datetime import datetime, timedelta
import wandb
import json
import pickle
import redis
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping of friendly names to actual column names
METRIC_MAPPING = {
    "accuracy": "eval.accuracy.true_fraction",
    "f1": "eval.f1",
    "precision": "eval.precision",
    "recall": "eval.recall",
    "hallucination_rate": "eval.hallucination_rate",
    "train_loss": "train.loss",
    "eval_loss": "eval.loss",
    "train_runtime": "train.runtime",
    "eval_runtime": "eval.runtime",
}

REQUIRED_COLUMNS = [
    "run_id",
    "run_name",
    "run_state",
    "created_at",
    "config.optim",
    "config.top_k",
    "config.top_p",
    "config.adam_beta1",
    "config.adam_beta2",
    "config.hidden_act",
    "config.max_length",
    "config.min_length",
    "config.model_type",
    "config.vocab_size",
    "config.hidden_size",
    "config.temperature",
    "config.hub_model_id",
    "config.rms_norm_eps",
    "config.weight_decay",
    "config.architectures",
    "config.learning_rate",
    "config.max_seq_length",
    "config.num_train_epochs",
    "config.initializer_range",
    "config.intermediate_size",
    "config.lr_scheduler_type",
    "config.num_hidden_layers",
    "config.num_attention_heads",
    "config.num_key_value_heads",
    "config.model.num_parameters",
    "config.max_position_embeddings",
    "config.tokenizer_name",
    "config.attn_implementation",
    "config.head_dim",
    "config.dataset_name",
] + list(
    METRIC_MAPPING.values()
)  # Add all metrics to required columns


class MetricCache:
    def __init__(self, data: Any, timestamp: datetime, ttl: int = 300):
        self.data = data
        self.timestamp = timestamp
        self.ttl = ttl

    def is_valid(self) -> bool:
        return datetime.now() - self.timestamp < timedelta(seconds=self.ttl)


class HallucinationDataManager:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port)
            self.redis_client.ping()
            logger.info("Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory-only cache.")
            self.redis_client = None

        self._memory_cache = {}
        self.df = None

    def get_all_data(self) -> pd.DataFrame:
        """Get all data as a DataFrame"""
        return self.df.copy() if self.df is not None else pd.DataFrame()

    def get_missing_columns(self) -> List[str]:
        """Get list of required columns that are missing"""
        if self.df is None:
            return REQUIRED_COLUMNS
        return list(set(REQUIRED_COLUMNS) - set(self.df.columns))

    async def _fetch_wandb_data(self) -> pd.DataFrame:
        """Fetch data from W&B hallucination project"""
        api = wandb.Api()
        logger.info(f"Fetching data from c-metrics/hallucination")

        runs = api.runs("c-metrics/hallucination")
        runs_list = list(runs)
        logger.info(f"Found {len(runs_list)} runs")

        def flatten_metrics(
            metrics: Dict[str, Any], prefix: str = ""
        ) -> Dict[str, Any]:
            """Flatten nested metrics dictionary"""
            flattened = {}
            for key, value in metrics.items():
                if not isinstance(value, dict):
                    metric_name = f"{prefix}{key}" if prefix else key
                    flattened[metric_name] = value
                    continue

                if "value" in value:
                    metric_name = f"{prefix}{key}" if prefix else key
                    flattened[metric_name] = value["value"]
                else:
                    for subkey, subvalue in value.items():
                        metric_name = (
                            f"{prefix}{key}.{subkey}" if prefix else f"{key}.{subkey}"
                        )
                        if isinstance(subvalue, dict):
                            flattened.update(
                                flatten_metrics({subkey: subvalue}, f"{prefix}{key}.")
                            )
                        else:
                            flattened[metric_name] = subvalue
            return flattened

        def parse_nested_value(value: Any) -> Dict[str, Any]:
            """Parse potentially nested values in strings"""
            if not isinstance(value, str):
                return {"value": value}

            try:
                parsed = json.loads(value.replace("'", '"'))
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except:
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"value": parsed}
                except:
                    return {"value": value}

        all_runs_data = []

        for run in runs_list:
            # Get basic run info
            run_data = {
                "run_id": run.id,
                "run_name": run.name,
                "run_state": run.state,
                "created_at": run.created_at,
            }

            # Add config parameters
            for key, value in run.config.items():
                parsed = parse_nested_value(value)
                if isinstance(parsed, dict) and "value" not in parsed:
                    flattened = flatten_metrics(parsed, f"config.{key}.")
                    run_data.update(flattened)
                else:
                    run_data[f"config.{key}"] = parsed.get("value", value)

            # Handle metrics from summary
            if hasattr(run, "summary"):
                metrics = {}

                # Handle standard metrics
                for key, value in run.summary._json_dict.items():
                    if key.startswith("_"):
                        continue

                    if key == "eval/accuracy":
                        try:
                            if isinstance(value, str):
                                parsed = ast.literal_eval(value)
                            else:
                                parsed = value
                            if isinstance(parsed, dict):
                                metrics["eval.accuracy.true_count"] = parsed.get(
                                    "true_count"
                                )
                                metrics["eval.accuracy.true_fraction"] = parsed.get(
                                    "true_fraction"
                                )
                                logger.info(
                                    f"Extracted accuracy for run {run.id}: {parsed}"
                                )
                            continue
                        except Exception as e:
                            logger.error(
                                f"Error parsing accuracy for run {run.id}: {str(e)}"
                            )
                            metrics[key.replace("/", ".")] = value
                    else:
                        parsed = parse_nested_value(value)
                        if isinstance(parsed, dict) and "value" not in parsed:
                            metrics.update(
                                flatten_metrics(parsed, f'{key.replace("/", ".")}.')
                            )
                        else:
                            metrics[key.replace("/", ".")] = parsed.get("value", value)

                run_data.update(metrics)

            all_runs_data.append(run_data)

        # Create DataFrame
        df = pd.DataFrame(all_runs_data)

        # Sort by creation date
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)

        # Verify accuracy data
        if "eval.accuracy.true_fraction" in df.columns:
            logger.info("\nAccuracy data extracted successfully:")
            logger.info(
                f"Number of accuracy values: {df['eval.accuracy.true_fraction'].count()}"
            )
            logger.info(f"Sample values:\n{df['eval.accuracy.true_fraction'].head()}")
        else:
            logger.error("Failed to extract accuracy data to DataFrame!")

        return df

    async def initialize(self) -> bool:
        """Initialize the data manager with W&B data"""
        try:
            # Fetch data from W&B
            self.df = await self._fetch_wandb_data()

            # Log available columns
            available_columns = set(self.df.columns)
            missing_columns = set(REQUIRED_COLUMNS) - available_columns
            if missing_columns:
                logger.warning(f"Missing columns in data: {missing_columns}")

            # Keep only required columns that are available
            columns_to_keep = [
                col for col in REQUIRED_COLUMNS if col in available_columns
            ]
            self.df = self.df[columns_to_keep]

            logger.info(
                f"Loaded DataFrame with {len(self.df)} rows and {len(columns_to_keep)} columns"
            )
            logger.info(f"Available columns: {columns_to_keep}")

            # Initialize basic caches
            await self._init_caches()

            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def get_best_models(self, metric: str = "accuracy", top_k: int = 5) -> List[Dict]:
        """Get best performing models for a metric"""
        try:
            # Map friendly metric name to actual column name
            metric_col = self._get_metric_column(metric)
            logger.info(f"Looking for best models using metric: {metric_col}")

            if metric_col not in self.df.columns:
                logger.warning(
                    f"Metric {metric_col} not found in data. Available metrics: {[col for col in self.df.columns if col.startswith('eval.') or col.startswith('train.')]}"
                )
                return []

            columns = [
                "run_id",
                "config.model_type",
                "config.learning_rate",
                metric_col,
            ]
            available_cols = [col for col in columns if col in self.df.columns]

            # Get best models
            best_models = self.df.nlargest(top_k, metric_col)[available_cols].to_dict(
                "records"
            )

            # Convert back to friendly names in the response
            reverse_mapping = {v: k for k, v in METRIC_MAPPING.items()}
            for model in best_models:
                renamed_model = {}
                for key, value in model.items():
                    friendly_name = reverse_mapping.get(key, key)
                    renamed_model[friendly_name] = value
                model.update(renamed_model)

            return best_models

        except Exception as e:
            logger.error(f"Error getting best models: {e}")
            return []

    def get_experiment_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific experiment"""
        try:
            run_data = self.df[self.df["run_id"] == run_id]
            if len(run_data) == 0:
                return None
            return run_data.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error getting experiment details: {e}")
            return None

    def compare_hyperparams(
        self, metric: str = "eval.accuracy.true_fraction"
    ) -> Dict[str, Any]:
        """Compare hyperparameter performance"""
        try:
            comparisons = {}

            if "config.learning_rate" in self.df.columns:
                lr_stats = (
                    self.df.groupby("config.learning_rate")[metric]
                    .agg(["mean", "std", "count"])
                    .round(4)
                    .to_dict("index")
                )
                comparisons["learning_rate"] = lr_stats

            if "config.num_train_epochs" in self.df.columns:
                epoch_stats = (
                    self.df.groupby("config.num_train_epochs")[metric]
                    .agg(["mean", "std", "count"])
                    .round(4)
                    .to_dict("index")
                )
                comparisons["epochs"] = epoch_stats

            return comparisons
        except Exception as e:
            logger.error(f"Error comparing hyperparameters: {e}")
            return {}

    async def _init_caches(self):
        """Initialize basic caches"""
        try:
            # Cache simple aggregations
            cache_data = {
                "model_types": (
                    self.df["config.model_type"].unique().tolist()
                    if "config.model_type" in self.df.columns
                    else []
                ),
                "learning_rates": (
                    sorted(self.df["config.learning_rate"].dropna().unique().tolist())
                    if "config.learning_rate" in self.df.columns
                    else []
                ),
                "metrics_summary": self._compute_metrics_summary(),
            }

            # Store in memory cache
            for key, value in cache_data.items():
                self._memory_cache[key] = MetricCache(
                    data=value, timestamp=datetime.now()
                )

            # Store in Redis if available
            if self.redis_client:
                for key, value in cache_data.items():
                    try:
                        self.redis_client.setex(
                            key, 300, pickle.dumps(value)  # 5 minutes TTL
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache {key} in Redis: {e}")

        except Exception as e:
            logger.error(f"Error initializing caches: {e}")

    def _compute_metrics_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for key metrics"""
        metrics = [
            "eval.accuracy.true_fraction",
            "eval.f1",
            "eval.precision",
            "eval.recall",
        ]
        summary = {}

        for metric in metrics:
            if metric in self.df.columns:
                summary[metric] = {
                    "mean": self.df[metric].mean(),
                    "std": self.df[metric].std(),
                    "min": self.df[metric].min(),
                    "max": self.df[metric].max(),
                }

        return summary

    def _get_metric_column(self, metric: str) -> str:
        """Convert friendly metric name to actual column name"""
        if metric in self.df.columns:
            return metric
        return METRIC_MAPPING.get(metric, metric)
