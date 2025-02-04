# core/agent/tools.py
from typing import Dict, List, Any, Optional
import pandas as pd
import logging
import numpy as np
from src.services.data_manager import METRIC_MAPPING

logger = logging.getLogger(__name__)


class AnalysisTools:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.DEFAULT_METRICS = [
            "eval.f1",
            "eval.precision",
            "eval.recall",
            "eval.hallucination_rate",
        ]
        self.CONFIG_COLUMNS = [
            col for col in self.data_manager.df.columns if col.startswith("config.")
        ]

    def _get_metric_column(self, friendly_name: str) -> str:
        """Convert friendly metric name to actual column name"""
        # If it's already a column name, return it
        if friendly_name in self.data_manager.df.columns:
            return friendly_name
        # Otherwise, try to map it
        return METRIC_MAPPING.get(friendly_name, friendly_name)

    async def get_best_models(
        self, metric: str = "accuracy", top_k: int = 5
    ) -> Dict[str, Any]:
        """Get top performing models for a specific metric"""
        try:
            metric_col = self._get_metric_column(metric)
            best_models = self.data_manager.get_best_models(metric_col, top_k)

            if not best_models:
                return {
                    "status": "error",
                    "error": f"No data available for metric: {metric}",
                }

            return {"status": "success", "data": best_models, "metric": metric}
        except Exception as e:
            logger.error(f"Error in get_best_models: {e}")
            return {"status": "error", "error": str(e)}

    async def compare_hyperparams(self, metric: str = "accuracy") -> Dict[str, Any]:
        """Compare performance across different hyperparameter settings"""
        try:
            metric_col = self._get_metric_column(metric)
            comparisons = self.data_manager.compare_hyperparams(metric_col)
            return {"status": "success", "data": comparisons, "metric": metric}
        except Exception as e:
            logger.error(f"Error in compare_hyperparams: {e}")
            return {"status": "error", "error": str(e)}

    async def get_experiment_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific experiment"""
        try:
            details = self.data_manager.get_experiment_details(run_id)
            if details is None:
                return {"status": "error", "error": f"Run {run_id} not found"}

            # Convert metric columns to friendly names
            friendly_metrics = {}
            reverse_mapping = {v: k for k, v in METRIC_MAPPING.items()}

            for key, value in details.items():
                if key in reverse_mapping:
                    friendly_metrics[reverse_mapping[key]] = value
                else:
                    friendly_metrics[key] = value

            return {"status": "success", "data": friendly_metrics}
        except Exception as e:
            logger.error(f"Error in get_experiment_details: {e}")
            return {"status": "error", "error": str(e)}

    async def analyze_by_model_type(
        self, metrics: Optional[List[str]] = None, group_by: str = "config.model_type"
    ) -> Dict[str, Any]:
        try:
            df = self.data_manager.df
            metrics = metrics or self.DEFAULT_METRICS

            logger.info(f"Starting analysis with metrics: {metrics}")
            logger.info(f"Grouping by: {group_by}")

            # Validate group_by column exists
            if group_by not in df.columns:
                return {
                    "status": "error",
                    "error": f"Group by column '{group_by}' not found",
                }

            # Validate metrics exist
            invalid_metrics = [m for m in metrics if m not in df.columns]
            if invalid_metrics:
                return {
                    "status": "error",
                    "error": f"Invalid metrics: {invalid_metrics}",
                }

            logger.info(f"Performing groupby operation on {len(df)} rows")

            # Perform groupby with correct aggregation syntax
            grouped_stats = df.groupby(group_by)[metrics].agg(
                [
                    "mean",
                    "std",
                    "count",
                    lambda x: x.quantile(0.25).round(4),
                    lambda x: x.quantile(0.75).round(4),
                ]
            )
            grouped_stats = grouped_stats.round(4)

            # Rename the lambda columns to something more readable
            grouped_stats = grouped_stats.rename(
                columns={"<lambda_0>": "q25", "<lambda_1>": "q75"}
            )

            # Convert to dictionary
            stats_dict = {}
            for model_type in grouped_stats.index:
                stats_dict[model_type] = {}
                for metric in metrics:
                    stats_dict[model_type][metric] = {
                        stat: float(grouped_stats.loc[model_type, (metric, stat)])
                        for stat in ["mean", "std", "count", "q25", "q75"]
                        if pd.notnull(grouped_stats.loc[model_type, (metric, stat)])
                    }

            # Calculate proportions
            total_runs = len(df)
            proportions = (
                df[group_by]
                .value_counts()
                .apply(lambda x: float(x / total_runs * 100))
                .round(2)
            )

            result = {
                "status": "success",
                "data": {
                    "statistics": stats_dict,
                    "proportions": proportions.to_dict(),
                    "total_runs": total_runs,
                },
            }

            logger.info("Analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in analyze_by_model_type: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def analyze_config_impact(
        self,
        target_metric: str = "eval.f1",
        config_params: Optional[List[str]] = None,
        method: str = "correlation",
    ) -> Dict[str, Any]:
        """
        Analyze how different configurations impact model performance.

        Args:
            target_metric: Metric to analyze impact on
            config_params: List of config parameters to analyze. If None, uses all numeric configs
            method: Analysis method ('correlation' or 'feature_importance')

        Returns:
            Dictionary with impact analysis results
        """
        try:
            df = self.data_manager.df

            # Validate target metric
            if target_metric not in df.columns:
                return {
                    "status": "error",
                    "error": f"Invalid target metric: {target_metric}",
                }

            # Get numeric config columns if not specified
            if config_params is None:
                config_params = [
                    col
                    for col in self.CONFIG_COLUMNS
                    if pd.api.types.is_numeric_dtype(df[col])
                ]

            # Calculate correlations
            correlations = {}
            for param in config_params:
                if param in df.columns and pd.api.types.is_numeric_dtype(df[param]):
                    corr = df[param].corr(df[target_metric])
                    if not pd.isna(corr):
                        correlations[param] = round(corr, 4)

            # Sort by absolute correlation value
            correlations = dict(
                sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            )

            # Calculate basic stats for each parameter
            param_stats = {}
            for param in config_params:
                if param in df.columns:
                    param_stats[param] = {
                        "mean": (
                            df[param].mean()
                            if pd.api.types.is_numeric_dtype(df[param])
                            else None
                        ),
                        "unique_values": df[param].nunique(),
                        "most_common": (
                            df[param].mode().iloc[0]
                            if len(df[param].mode()) > 0
                            else None
                        ),
                    }

            return {
                "status": "success",
                "data": {
                    "correlations": correlations,
                    "parameter_stats": param_stats,
                    "target_metric": target_metric,
                },
            }
        except Exception as e:
            logger.error(f"Error in analyze_config_impact: {e}")
            return {"status": "error", "error": str(e)}

    async def query_data(
        self,
        filters: Dict[str, Any] = None,
        columns: List[str] = None,
        sort_by: str = None,
        ascending: bool = True,
        limit: int = 10,
        group_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Query data with filters and return selected rows with optional grouping statistics.

        Args:
            filters: Dictionary of column:value pairs for filtering.
                    Values can be single values or lists/ranges.
                    Example: {
                        'config.model_type': 'llama',
                        'eval.f1': {'min': 0.5, 'max': 0.8},
                        'config.learning_rate': [0.001, 0.0001]
                    }
            columns: List of columns to return. If None, returns all columns
            sort_by: Column to sort by
            ascending: Sort order
            limit: Maximum number of rows to return
            group_stats: If True, includes basic statistics for numeric columns

        Returns:
            Dictionary containing filtered data and optional statistics
        """
        try:
            df = self.data_manager.df.copy()

            # Apply filters
            if filters:
                for col, condition in filters.items():
                    if col not in df.columns:
                        return {"status": "error", "error": f"Column {col} not found"}

                    if isinstance(condition, dict):
                        # Handle range conditions
                        if "min" in condition:
                            df = df[df[col] >= condition["min"]]
                        if "max" in condition:
                            df = df[df[col] <= condition["max"]]
                    elif isinstance(condition, (list, tuple)):
                        # Handle list of values
                        df = df[df[col].isin(condition)]
                    else:
                        # Handle single value
                        df = df[df[col] == condition]

            if columns:
                invalid_cols = [col for col in columns if col not in df.columns]
                if invalid_cols:
                    return {
                        "status": "error",
                        "error": f"Invalid columns: {invalid_cols}",
                    }
                df = df[columns]

            if sort_by:
                if sort_by not in df.columns:
                    return {
                        "status": "error",
                        "error": f"Sort column {sort_by} not found",
                    }
                df = df.sort_values(by=sort_by, ascending=ascending)

            # calculate group stats if needed
            stats = None
            if group_stats:
                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                stats = {}

                for col in numeric_cols:
                    col_stats = df[col].describe()
                    stats[col] = {
                        "mean": float(col_stats["mean"]),
                        "std": float(col_stats["std"]),
                        "min": float(col_stats["min"]),
                        "max": float(col_stats["max"]),
                        "25%": float(col_stats["25%"]),
                        "50%": float(col_stats["50%"]),
                        "75%": float(col_stats["75%"]),
                        "count": int(col_stats["count"]),
                    }

            # Convert dtf to dict format, handling NaN values
            filtered_data = df.head(limit).replace({np.nan: None}).to_dict("records")

            return {
                "status": "success",
                "data": {
                    "total_matches": len(df),
                    "returned_rows": len(filtered_data),
                    "rows": filtered_data,
                    "statistics": stats if group_stats else None,
                },
            }

        except Exception as e:
            logger.error(f"Error in query_data: {e}")
            return {"status": "error", "error": str(e)}

    async def get_performance_distribution(
        self,
        metrics: Optional[List[str]] = None,
        percentiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    ) -> Dict[str, Any]:
        """
        Get statistical distribution of performance metrics.

        Args:
            metrics: List of metrics to analyze. If None, uses default metrics
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary with distribution statistics
        """
        try:
            df = self.data_manager.df
            metrics = metrics or self.DEFAULT_METRICS

            # Validate metrics
            invalid_metrics = [m for m in metrics if m not in df.columns]
            if invalid_metrics:
                return {
                    "status": "error",
                    "error": f"Invalid metrics: {invalid_metrics}",
                }

            distributions = {}
            for metric in metrics:
                values = df[metric].dropna()

                distributions[metric] = {
                    "mean": round(values.mean(), 4),
                    "std": round(values.std(), 4),
                    "min": round(values.min(), 4),
                    "max": round(values.max(), 4),
                    "percentiles": {
                        str(int(p * 100)): round(values.quantile(p), 4)
                        for p in percentiles
                    },
                }

            return {"status": "success", "data": distributions}
        except Exception as e:
            logger.error(f"Error in get_performance_distribution: {e}")
            return {"status": "error", "error": str(e)}

    async def compare_architectures(
        self,
        metrics: Optional[List[str]] = None,
        arch_column: str = "config.architectures",
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        try:
            df = self.data_manager.df.copy()
            metrics = metrics or self.DEFAULT_METRICS

            logger.info(f"Sample of raw values:")
            logger.info(df[arch_column].head().to_dict())

            def clean_architecture(x):
                # used to clean the architecture column
                if pd.isna(x):
                    return "Unknown"
                try:
                    cleaned = str(x).strip("\"[]'").split(",")[0].strip()
                    return cleaned if cleaned else "Unknown"
                except:
                    return "Unknown"

            df[arch_column] = df[arch_column].apply(clean_architecture)

            logger.info(
                f"Original architecture values:\n{self.data_manager.df[arch_column].value_counts()}"
            )
            logger.info(
                f"Cleaned architecture values:\n{df[arch_column].value_counts()}"
            )

            # Validate metrics
            invalid_metrics = [m for m in metrics if m not in df.columns]
            if invalid_metrics:
                return {
                    "status": "error",
                    "error": f"Invalid metrics: {invalid_metrics}",
                }

            # Get architectures with sufficient samples
            arch_counts = df[arch_column].value_counts()
            valid_archs = arch_counts[arch_counts >= min_samples].index

            logger.info(
                f"Valid architectures (>={min_samples} samples): {valid_archs.tolist()}"
            )

            if len(valid_archs) < 2:
                return {
                    "status": "error",
                    "error": f"Need at least 2 architectures with {min_samples}+ samples",
                }

            results = {}
            for metric in metrics:
                arch_stats = {}
                statistical_tests = {}

                # Calculate basic stats for each architecture
                for arch in valid_archs:
                    values = df[df[arch_column] == arch][metric].dropna()
                    logger.info(
                        f"Architecture {arch} has {len(values)} samples for {metric}"
                    )

                    if len(values) == 0:
                        continue

                    arch_stats[arch] = {
                        "mean": round(float(values.mean()), 4),
                        "std": round(float(values.std()), 4),
                        "count": len(values),
                    }

                # Perform statistical tests between architectures
                for i, arch1 in enumerate(valid_archs):
                    for arch2 in valid_archs[i + 1 :]:
                        values1 = df[df[arch_column] == arch1][metric].dropna()
                        values2 = df[df[arch_column] == arch2][metric].dropna()

                        logger.info(f"Comparing {arch1} vs {arch2} for {metric}")
                        logger.info(f"Samples: {len(values1)} vs {len(values2)}")

                        if len(values1) < 2 or len(values2) < 2:
                            statistical_tests[f"{arch1}_vs_{arch2}"] = {
                                "error": "Insufficient samples for comparison"
                            }
                            continue

                        try:
                            from scipy import stats

                            t_stat, p_value = stats.ttest_ind(values1, values2)
                            statistical_tests[f"{arch1}_vs_{arch2}"] = {
                                "t_statistic": round(float(t_stat), 4),
                                "p_value": round(float(p_value), 4),
                                "significant": float(p_value) < 0.05,
                            }
                        except Exception as e:
                            logger.error(
                                f"T-test failed for {arch1} vs {arch2}: {str(e)}"
                            )
                            statistical_tests[f"{arch1}_vs_{arch2}"] = {"error": str(e)}

                results[metric] = {
                    "architecture_stats": arch_stats,
                    "statistical_tests": statistical_tests,
                }

            return {"status": "success", "data": results}

        except Exception as e:
            logger.error(f"Error in compare_architectures: {e}")
            return {"status": "error", "error": str(e)}


# Available tools for the LLM agent
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_best_models",
            "description": "Get top performing models for a specific metric",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to rank by (e.g., accuracy, f1, precision, recall)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top models to return",
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_hyperparams",
            "description": "Compare performance across different hyperparameter settings",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to use for comparison (e.g., accuracy, f1)",
                    }
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_details",
            "description": "Get detailed information about a specific experiment",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "ID of the experiment run",
                    }
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_by_model_type",
            "description": "Get performance statistics grouped by model type, including means, distributions, and proportions of experiments",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to analyze (e.g., ['eval.f1', 'eval.precision']). If not provided, uses default metrics",
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Column to group by (default: config.model_type)",
                        "default": "config.model_type",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_config_impact",
            "description": "Analyze how different configuration parameters impact model performance using correlation analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_metric": {
                        "type": "string",
                        "description": "Target metric to analyze impact on (e.g., eval.f1)",
                        "default": "eval.f1",
                    },
                    "config_params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of config parameters to analyze. If not provided, uses all numeric configs",
                    },
                    "method": {
                        "type": "string",
                        "description": "Analysis method ('correlation' or 'feature_importance')",
                        "enum": ["correlation", "feature_importance"],
                        "default": "correlation",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_performance_distribution",
            "description": "Get statistical distribution of performance metrics including means, standard deviations, and percentiles",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to analyze. If not provided, uses default metrics",
                    },
                    "percentiles": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of percentiles to calculate (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])",
                        "default": [0.1, 0.25, 0.5, 0.75, 0.9],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_architectures",
            "description": "Compare different model architectures with statistical tests and performance metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to compare. If not provided, uses default metrics",
                    },
                    "arch_column": {
                        "type": "string",
                        "description": "Column containing architecture information",
                        "default": "config.architectures",
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "Minimum samples required for comparison",
                        "default": 5,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "Query and filter experiment data with custom conditions. Available metrics include both friendly names and column names:\n\nMetric Mappings:\n- accuracy → eval.accuracy.true_fraction\n- f1 → eval.f1\n- precision → eval.precision\n- recall → eval.recall\n- hallucination_rate → eval.hallucination_rate\n- train_loss → train.loss\n- eval_loss → eval.loss\n- train_runtime → train.runtime\n- eval_runtime → eval.runtime\n\nConfiguration columns available:\n- Model: config.model_type, config.architectures, config.hub_model_id\n- Architecture: config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads, config.head_dim\n- Training: config.learning_rate, config.num_train_epochs, config.lr_scheduler_type, config.optim\n- Sequence: config.max_length, config.min_length, config.max_seq_length, config.max_position_embeddings\n- Other: config.temperature, config.top_k, config.top_p, config.dataset_name",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": "Dictionary of column:value pairs for filtering. Can use either friendly names (e.g., 'f1') or column names (e.g., 'eval.f1'). Values can be single values or dictionaries with 'min'/'max' for ranges",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to return. Can use friendly names or column names. If not provided, returns key columns",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Column to sort results by (can use friendly name or column name)",
                    },
                    "ascending": {
                        "type": "boolean",
                        "description": "Sort order (true for ascending, false for descending)",
                        "default": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return",
                        "default": 10,
                    },
                    "group_stats": {
                        "type": "boolean",
                        "description": "Include basic statistics for numeric columns",
                        "default": False,
                    },
                },
            },
        },
    },
]
