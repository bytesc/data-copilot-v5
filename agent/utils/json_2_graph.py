import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

matplotlib.use('Agg')  # Use non-interactive backend


def visualize_opsearch_results(
        query_json: Dict[str, Any],
        result_json: Dict[str, Any],
        output_dir: str = "./"
) -> str:
    """
    Visualize OpenSearch query results based on query and result JSON.

    Args:
        query_json: The query JSON that was executed
        result_json: The result JSON from OpenSearch
        output_dir: Directory to save the visualization image

    Returns:
        Path to the saved image file
    """

    def _create_output_path(chart_type: str) -> str:
        """Create unique output file path."""
        import time
        timestamp = int(time.time())
        filename = f"opsearch_chart_{chart_type}_{timestamp}.png"
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _extract_query_type() -> str:
        """Extract query type from query JSON."""
        return query_json.get("query", {}).get("type", "").lower()

    def _extract_fields_from_query() -> List[str]:
        """Extract relevant fields from query configuration."""
        config = query_json.get("query", {}).get("config", {})
        fields = config.get("fields", [])
        dimensions = config.get("dimensions", [])
        groups = config.get("groups", [])
        return list(set(fields + dimensions + groups))

    def _extract_axis_labels() -> Dict[str, str]:
        """Extract axis labels from query JSON."""
        config = query_json.get("query", {}).get("config", {})

        # Default labels
        x_label = "Categories"
        y_label = "Count/Value"

        # Try to extract from dimensions
        dimensions = config.get("dimensions", [])
        if dimensions:
            x_label = dimensions[0].replace("_", " ").title()

        # Try to extract from buckets
        buckets = config.get("buckets", [])
        if buckets:
            for bucket in buckets:
                if bucket.get("type") == "range":
                    x_label = bucket.get("field", x_label).replace("_", " ").title()
                    break
                elif bucket.get("type") == "date_histogram":
                    x_label = "Time Period"
                    break
                elif bucket.get("type") == "terms":
                    x_label = bucket.get("field", x_label).replace("_", " ").title()
                    break

        # Try to extract from metrics_field or fields
        metrics_field = config.get("metrics_field")
        if metrics_field:
            y_label = metrics_field.replace("_", " ").title()
        elif config.get("fields"):
            y_label = config["fields"][0].replace("_", " ").title()

        return {"x_label": x_label, "y_label": y_label}

    def _determine_chart_type(query_type: str, data_structure: Dict) -> str:
        """Determine the most appropriate chart type based on data."""

        if query_type == "stats":
            # For statistical calculations, use summary charts
            if len(data_structure) > 1:
                return "multi_metric_summary"
            return "single_metric_summary"

        elif query_type == "distribution":
            # For distribution analysis
            if "buckets" in data_structure:
                buckets = data_structure.get("buckets", [])
                if buckets and isinstance(buckets, list):
                    # Check for date histogram
                    config = query_json.get("query", {}).get("config", {})
                    if config.get("buckets"):
                        for bucket in config["buckets"]:
                            if bucket.get("type") == "date_histogram":
                                return "time_series"

                    # Check for multiple dimensions
                    if buckets and "dimensions" in buckets[0]:
                        first_bucket = buckets[0]
                        dimensions = first_bucket.get("dimensions", {})
                        if dimensions and len(dimensions) > 1:
                            return "stacked_bar"
                        return "grouped_bar"

                    # Check for range buckets
                    if "from" in buckets[0] or "to" in buckets[0]:
                        return "histogram"

                    # Check for grouped data
                    if "groups" in buckets[0]:
                        return "grouped_bar"

                    return "bar"

            # Check for percentage data
            if any("%" in str(key).lower() or "percentage" in str(key).lower()
                   for key in data_structure.keys()):
                return "pie"

        return "bar"  # Default fallback

    def _flatten_distribution_data(data: Dict) -> pd.DataFrame:
        """Flatten nested distribution data into a DataFrame."""
        flattened_data = []

        def _extract_metrics(metrics_dict: Dict, prefix: str = "") -> Dict:
            """Extract metrics from a dictionary."""
            extracted = {}
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    extracted.update(_extract_metrics(value, f"{prefix}{key}_"))
                else:
                    extracted[f"{prefix}{key}"] = value
            return extracted

        if "buckets" in data:
            for bucket in data["buckets"]:
                row = {}

                # Add bucket key
                if "key" in bucket:
                    row["bucket_key"] = bucket["key"]
                if "from" in bucket:
                    row["from"] = bucket["from"]
                if "to" in bucket:
                    row["to"] = bucket["to"]

                # Add metrics
                if "metrics" in bucket:
                    row.update(_extract_metrics(bucket["metrics"], "metrics_"))
                elif "doc_count" in bucket:
                    row["doc_count"] = bucket["doc_count"]

                # Add dimensions
                if "dimensions" in bucket:
                    for dim_name, dim_data in bucket["dimensions"].items():
                        if "buckets" in dim_data:
                            for dim_bucket in dim_data["buckets"]:
                                dim_row = row.copy()
                                dim_row[f"dim_{dim_name}"] = dim_bucket.get("key", "")
                                if "doc_count" in dim_bucket:
                                    dim_row[f"dim_{dim_name}_count"] = dim_bucket["doc_count"]
                                if "percentage" in dim_bucket:
                                    dim_row[f"dim_{dim_name}_percentage"] = dim_bucket["percentage"]
                                flattened_data.append(dim_row)
                else:
                    flattened_data.append(row)

        return pd.DataFrame(flattened_data)

    def _plot_stats_summary(data: Dict, output_path: str) -> str:
        """Plot statistical summary charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Statistical Summary Analysis", fontsize=16, fontweight='bold')

        # Convert to DataFrame for easier handling
        stats_df = pd.DataFrame(data).T

        # 1. Summary Metrics Bar Chart
        ax1 = axes[0, 0]
        metrics_to_plot = ['min', 'max', 'avg', 'median', 'q1', 'q3']
        available_metrics = [m for m in metrics_to_plot if m in stats_df.columns]

        if available_metrics:
            metric_data = stats_df[available_metrics]
            x = np.arange(len(stats_df))
            width = 0.15

            for i, metric in enumerate(available_metrics):
                values = metric_data[metric].values
                ax1.bar(x + i * width - width * (len(available_metrics) - 1) / 2,
                        values, width, label=metric.replace('_', ' ').title())

            ax1.set_xlabel('Fields')
            ax1.set_ylabel('Values')
            ax1.set_title('Key Statistics by Field')
            ax1.set_xticks(x)
            ax1.set_xticklabels(stats_df.index, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Count Chart
        ax2 = axes[0, 1]
        if 'count' in stats_df.columns:
            ax2.bar(stats_df.index, stats_df['count'])
            ax2.set_xlabel('Fields')
            ax2.set_ylabel('Count')
            ax2.set_title('Document Count by Field')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for i, v in enumerate(stats_df['count']):
                ax2.text(i, v, str(int(v)), ha='center', va='bottom')

        # 3. Spread Indicators
        ax3 = axes[1, 0]
        spread_metrics = ['std_deviation', 'variance']
        available_spread = [m for m in spread_metrics if m in stats_df.columns]

        if available_spread:
            spread_data = stats_df[available_spread]
            x = np.arange(len(stats_df))
            width = 0.35

            for i, metric in enumerate(available_spread):
                values = spread_data[metric].values
                ax3.bar(x + i * width - width / 2, values, width,
                        label=metric.replace('_', ' ').title())

            ax3.set_xlabel('Fields')
            ax3.set_ylabel('Spread Value')
            ax3.set_title('Data Spread Indicators')
            ax3.set_xticks(x)
            ax3.set_xticklabels(stats_df.index, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        # Prepare table data
        table_data = []
        for field in stats_df.index:
            row = [field]
            for metric in ['count', 'min', 'max', 'avg', 'median']:
                if metric in stats_df.columns:
                    value = stats_df.loc[field, metric]
                    if isinstance(value, (int, np.integer)):
                        row.append(f"{value:,}")
                    elif isinstance(value, float):
                        row.append(f"{value:.2f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("")
            table_data.append(row)

        columns = ['Field', 'Count', 'Min', 'Max', 'Avg', 'Median']
        table = ax4.table(cellText=table_data, colLabels=columns,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Statistical Summary', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def _plot_distribution_bar(data: Dict, output_path: str) -> str:
        """Plot bar chart for distribution data."""
        # Extract axis labels from query JSON
        axis_labels = _extract_axis_labels()

        df = _flatten_distribution_data(data)

        if df.empty:
            # Handle simple dictionary data
            if isinstance(data, dict) and not any(isinstance(v, dict) for v in data.values()):
                fig, ax = plt.subplots(figsize=(12, 8))
                keys = list(data.keys())
                values = list(data.values())

                bars = ax.bar(range(len(keys)), values, color='steelblue', alpha=0.7)
                ax.set_xlabel(axis_labels['x_label'])
                ax.set_ylabel(axis_labels['y_label'])
                ax.set_title('Distribution Analysis')
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels(keys, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.0f}', ha='center', va='bottom')
            else:
                # Try to extract from buckets
                buckets = data.get('buckets', [])
                if buckets:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    labels = [str(b.get('key', f'Bucket {i}')) for i, b in enumerate(buckets)]
                    counts = [b.get('doc_count', b.get('metrics', {}).get('count', 0))
                              for b in buckets]

                    bars = ax.bar(labels, counts, color='steelblue', alpha=0.7)
                    ax.set_xlabel(axis_labels['x_label'])
                    ax.set_ylabel(axis_labels['y_label'])
                    ax.set_title('Distribution by Bucket')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{height:.0f}', ha='center', va='bottom')
        else:
            # Plot from flattened DataFrame
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Plot 1: Main distribution
            if 'bucket_key' in df.columns:
                ax1 = axes[0]
                if 'doc_count' in df.columns:
                    df_grouped = df.groupby('bucket_key')['doc_count'].sum().reset_index()
                    bars = ax1.bar(df_grouped['bucket_key'], df_grouped['doc_count'],
                                   color='steelblue', alpha=0.7)
                    ax1.set_ylabel(axis_labels['y_label'])
                elif 'metrics_count' in df.columns:
                    df_grouped = df.groupby('bucket_key')['metrics_count'].sum().reset_index()
                    bars = ax1.bar(df_grouped['bucket_key'], df_grouped['metrics_count'],
                                   color='steelblue', alpha=0.7)
                    ax1.set_ylabel(axis_labels['y_label'])

                ax1.set_xlabel(axis_labels['x_label'])
                ax1.set_title('Distribution by Bucket')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.0f}', ha='center', va='bottom')

            # Plot 2: Sub-distribution if available
            ax2 = axes[1]
            dim_cols = [col for col in df.columns if
                        col.startswith('dim_') and not col.endswith('_count') and not col.endswith('_percentage')]

            if dim_cols:
                for dim_col in dim_cols[:3]:  # Limit to 3 dimensions
                    count_col = f"{dim_col}_count"
                    if count_col in df.columns:
                        dim_data = df.groupby(dim_col)[count_col].sum().reset_index()
                        if len(dim_data) > 0:
                            ax2.bar(dim_data[dim_col], dim_data[count_col],
                                    alpha=0.6, label=dim_col.replace('dim_', ''))

                ax2.set_xlabel('Dimension Values')
                ax2.set_ylabel(axis_labels['y_label'])
                ax2.set_title('Distribution by Dimension')
                ax2.legend()
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            else:
                # Plot percentage if available
                perc_cols = [col for col in df.columns if 'percentage' in col.lower()]
                if perc_cols:
                    for perc_col in perc_cols[:2]:
                        if 'bucket_key' in df.columns:
                            ax2.bar(df['bucket_key'], df[perc_col],
                                    alpha=0.6, label=perc_col.replace('metrics_', '').replace('_', ' '))
                    ax2.set_xlabel(axis_labels['x_label'])
                    ax2.set_ylabel('Percentage')
                    ax2.set_title('Percentage Distribution')
                    ax2.legend()
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
                else:
                    axes[1].axis('off')

        plt.suptitle("Distribution Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def _plot_grouped_distribution(data: Dict, output_path: str) -> str:
        """Plot grouped distribution analysis."""
        # Extract axis labels from query JSON
        axis_labels = _extract_axis_labels()

        df = _flatten_distribution_data(data)

        if df.empty or 'bucket_key' not in df.columns:
            return _plot_distribution_bar(data, output_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Find dimension columns
        dim_cols = [col for col in df.columns
                    if col.startswith('dim_') and not col.endswith('_count') and not col.endswith('_percentage')]

        if not dim_cols:
            return _plot_distribution_bar(data, output_path)

        # Plot 1: Grouped bar chart for first dimension
        ax1 = axes[0]
        first_dim = dim_cols[0]
        count_col = f"{first_dim}_count"

        if count_col in df.columns:
            # Create pivot table
            pivot_data = df.pivot_table(
                values=count_col,
                index='bucket_key',
                columns=first_dim,
                aggfunc='sum',
                fill_value=0
            )

            x = np.arange(len(pivot_data))
            width = 0.8 / len(pivot_data.columns)

            for i, col in enumerate(pivot_data.columns):
                ax1.bar(x + i * width - width * (len(pivot_data.columns) - 1) / 2,
                        pivot_data[col].values, width, label=str(col))

            ax1.set_xlabel(axis_labels['x_label'])
            ax1.set_ylabel(axis_labels['y_label'])
            ax1.set_title(f'Distribution by {first_dim.replace("dim_", "")} and Bucket')
            ax1.set_xticks(x)
            ax1.set_xticklabels(pivot_data.index, rotation=45, ha='right')
            ax1.legend(title=first_dim.replace('dim_', ''))
            ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Stacked bar or pie chart
        ax2 = axes[1]

        if len(dim_cols) > 1:
            # Stacked bar for multiple dimensions
            second_dim = dim_cols[1] if len(dim_cols) > 1 else first_dim
            second_count_col = f"{second_dim}_count"

            if second_count_col in df.columns:
                stacked_data = df.pivot_table(
                    values=second_count_col,
                    index='bucket_key',
                    columns=second_dim,
                    aggfunc='sum',
                    fill_value=0
                )

                bottom = np.zeros(len(stacked_data))
                for col in stacked_data.columns:
                    ax2.bar(stacked_data.index, stacked_data[col],
                            bottom=bottom, label=str(col))
                    bottom += stacked_data[col].values

                ax2.set_xlabel(axis_labels['x_label'])
                ax2.set_ylabel(axis_labels['y_label'])
                ax2.set_title(f'Stacked Distribution by {second_dim.replace("dim_", "")}')
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend(title=second_dim.replace('dim_', ''))
                ax2.grid(True, alpha=0.3, axis='y')
        else:
            # Pie chart for single dimension distribution
            if 'doc_count' in df.columns or 'metrics_count' in df.columns:
                count_col = 'doc_count' if 'doc_count' in df.columns else 'metrics_count'
                dim_counts = df.groupby(first_dim)[count_col].sum()

                if len(dim_counts) <= 10:  # Limit pie chart to 10 segments
                    wedges, texts, autotexts = ax2.pie(
                        dim_counts.values,
                        labels=dim_counts.index,
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax2.set_title(f'Percentage Distribution by {first_dim.replace("dim_", "")}')
                else:
                    # Too many segments, use bar chart
                    ax2.bar(range(len(dim_counts)), dim_counts.values)
                    ax2.set_xlabel(first_dim.replace('dim_', ''))
                    ax2.set_ylabel(axis_labels['y_label'])
                    ax2.set_title(f'Distribution by {first_dim.replace("dim_", "")}')
                    ax2.set_xticks(range(len(dim_counts)))
                    ax2.set_xticklabels(dim_counts.index, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3)

        plt.suptitle("Grouped Distribution Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def _plot_histogram(data: Dict, output_path: str) -> str:
        """Plot histogram for range bucket data."""
        # Extract axis labels from query JSON
        axis_labels = _extract_axis_labels()

        buckets = data.get('buckets', [])

        if not buckets:
            return _plot_distribution_bar(data, output_path)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract bin edges and counts
        bin_edges = []
        counts = []
        labels = []

        for bucket in buckets:
            from_val = bucket.get('from', 0)
            to_val = bucket.get('to', from_val + 1)  # Default width of 1
            count = bucket.get('doc_count', bucket.get('metrics', {}).get('count', 0))

            bin_edges.append(from_val)
            if bucket == buckets[-1]:  # Last bucket
                bin_edges.append(to_val if to_val is not None else from_val + 1)

            counts.append(count)
            labels.append(bucket.get('key', f'{from_val}-{to_val}'))

        # Create histogram
        bars = ax.bar(range(len(counts)), counts,
                      color='steelblue', alpha=0.7, width=0.8)

        ax.set_xlabel(axis_labels['x_label'])
        ax.set_ylabel(axis_labels['y_label'])
        ax.set_title('Histogram Distribution')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.0f}', ha='center', va='bottom')

        # Add cumulative line - REMOVED to eliminate line plot
        # if len(counts) > 1:
        #     ax2 = ax.twinx()
        #     cumulative = np.cumsum(counts)
        #     ax2.plot(range(len(cumulative)), cumulative, 'r-', marker='o',
        #              linewidth=2, markersize=6, label='Cumulative')
        #     ax2.set_ylabel('Cumulative Count', color='r')
        #     ax2.tick_params(axis='y', labelcolor='r')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def _plot_time_series(data: Dict, output_path: str) -> str:
        """Plot time series data."""
        # Extract axis labels from query JSON
        axis_labels = _extract_axis_labels()

        buckets = data.get('buckets', [])

        if not buckets:
            return _plot_distribution_bar(data, output_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Extract time series data
        timestamps = []
        counts = []
        metrics = {}

        for bucket in buckets:
            key = bucket.get('key', '')
            timestamps.append(key)
            count = bucket.get('doc_count', bucket.get('metrics', {}).get('count', 0))
            counts.append(count)

            # Extract other metrics
            bucket_metrics = bucket.get('metrics', {})
            for metric_name, metric_value in bucket_metrics.items():
                if metric_name != 'count' and metric_name != 'percentage':
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)

        # Plot 1: Count over time - CHANGED FROM LINE TO BAR CHART
        ax1 = axes[0]
        x_pos = range(len(timestamps))

        bars = ax1.bar(x_pos, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel(axis_labels['x_label'])
        ax1.set_ylabel(axis_labels['y_label'])
        ax1.set_title('Time Series - Count')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(timestamps, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Other metrics over time - CHANGED FROM LINE TO BAR CHART
        ax2 = axes[1]

        if metrics:
            # Use bar chart instead of line plot for metrics
            metric_names = list(metrics.keys())
            if metric_names:
                # Group bar chart for multiple metrics
                x = np.arange(len(timestamps))
                width = 0.8 / len(metric_names)

                for i, metric_name in enumerate(metric_names):
                    values = metrics[metric_name]
                    if len(values) == len(timestamps):
                        ax2.bar(x + i * width - width * (len(metric_names) - 1) / 2, values, width,
                                label=metric_name.replace('_', ' ').title())

                ax2.set_xlabel(axis_labels['x_label'])
                ax2.set_ylabel('Metric Value')
                ax2.set_title('Time Series - Metrics')
                ax2.set_xticks(x)
                ax2.set_xticklabels(timestamps, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        else:
            # Use bar chart for cumulative count instead of line plot
            cumulative = np.cumsum(counts)
            ax2.bar(x_pos, cumulative, color='lightcoral', alpha=0.7)
            ax2.set_xlabel(axis_labels['x_label'])
            ax2.set_ylabel('Cumulative Count')
            ax2.set_title('Cumulative Count Over Time')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(timestamps, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

        plt.suptitle("Time Series Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def _plot_pie_chart(data: Dict, output_path: str) -> str:
        """Plot pie chart for percentage data."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Try to extract percentage data
        if isinstance(data, dict):
            # Direct percentage dictionary
            if all(isinstance(v, (int, float)) for v in data.values()):
                labels = list(data.keys())
                sizes = list(data.values())

                ax1 = axes[0]
                wedges, texts, autotexts = ax1.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.85
                )

                # Draw circle for donut chart
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax1.add_artist(centre_circle)
                ax1.set_title('Percentage Distribution')

                # Bar chart of same data
                ax2 = axes[1]
                bars = ax2.bar(range(len(labels)), sizes, color='lightcoral', alpha=0.7)
                ax2.set_xlabel('Categories')
                ax2.set_ylabel('Percentage')
                ax2.set_title('Percentage by Category')
                ax2.set_xticks(range(len(labels)))
                ax2.set_xticklabels(labels, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.1f}%', ha='center', va='bottom')

        plt.suptitle("Percentage Distribution Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    # Main function logic
    try:
        # Determine chart type
        query_type = _extract_query_type()
        chart_type = _determine_chart_type(query_type, result_json)

        # Create output path
        output_path = _create_output_path(chart_type)

        # Select and call appropriate plot function
        if chart_type == "single_metric_summary" or chart_type == "multi_metric_summary":
            return _plot_stats_summary(result_json, output_path)

        elif chart_type == "time_series":
            return _plot_time_series(result_json, output_path)

        elif chart_type == "histogram":
            return _plot_histogram(result_json, output_path)

        elif chart_type == "grouped_bar" or chart_type == "stacked_bar":
            return _plot_grouped_distribution(result_json, output_path)

        elif chart_type == "pie":
            return _plot_pie_chart(result_json, output_path)

        else:  # Default to bar chart
            return _plot_distribution_bar(result_json, output_path)

    except Exception as e:
        # Create error visualization
        error_path = _create_output_path("error")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Visualization Error:\n{str(e)}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.5))
        ax.set_title("Error in Visualization", fontsize=14, fontweight='bold', color='red')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        plt.close()
        return error_path


if __name__ == "__main__":
    # Example 1: Basic statistical calculation (stats type)
    print("Example 1: Basic statistical calculation")
    query_stats = {
        "query": {
            "type": "stats",
            "config": {
                "fields": ["age", "salary", "blood_pressure", "cholesterol"],
                "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation", "count"],
                "filters": [
                    {
                        "field": "gender",
                        "operator": "eq",
                        "value": "male"
                    }
                ]
            }
        }
    }

    result_stats = {
        "age": {
            "count": 1000,
            "min": 18,
            "max": 80,
            "avg": 45.5,
            "median": 45.0,
            "q1": 30.0,
            "q3": 60.0,
            "std_deviation": 15.2
        },
        "salary": {
            "count": 1000,
            "min": 3000,
            "max": 50000,
            "avg": 15000.5,
            "median": 12000.0,
            "q1": 8000.0,
            "q3": 20000.0,
            "std_deviation": 8000.5
        },
        "blood_pressure": {
            "count": 800,
            "min": 90,
            "max": 180,
            "avg": 120.5,
            "median": 118.0,
            "q1": 110.0,
            "q3": 130.0,
            "std_deviation": 12.3
        },
        "cholesterol": {
            "count": 750,
            "min": 120,
            "max": 280,
            "avg": 195.2,
            "median": 192.0,
            "q1": 170.0,
            "q3": 215.0,
            "std_deviation": 25.8
        }
    }

    image_path1 = visualize_opsearch_results(query_stats, result_stats, "./examples")
    print(f"Stats chart saved to: {image_path1}")

    # Example 2: Simple distribution with percentage (gender distribution)
    print("\nExample 2: Gender distribution with percentages")
    query_gender = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["gender"],
                "metrics": ["count", "percentage"],
                "filters": [
                    {
                        "field": "has_diabetes",
                        "operator": "eq",
                        "value": True
                    }
                ]
            }
        }
    }

    result_gender = {
        "buckets": [
            {
                "key": "Male",
                "doc_count": 650,
                "metrics": {
                    "count": 650,
                    "percentage": 52.0
                }
            },
            {
                "key": "Female",
                "doc_count": 600,
                "metrics": {
                    "count": 600,
                    "percentage": 48.0
                }
            }
        ]
    }

    image_path2 = visualize_opsearch_results(query_gender, result_gender, "./examples")
    print(f"Gender distribution chart saved to: {image_path2}")

    # Example 3: Multi-dimensional analysis (disease by age and gender)
    print("\nExample 3: Multi-dimensional analysis - disease by age and gender")
    query_multi_dim = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["diabetic_retinopathy", "hypertension"],
                "groups": ["gender"],
                "buckets": [
                    {
                        "type": "range",
                        "field": "age",
                        "ranges": [
                            {"key": "<40", "from": 0, "to": 40},
                            {"key": "40-60", "from": 40, "to": 60},
                            {"key": "60+", "from": 60}
                        ]
                    }
                ],
                "metrics": ["count", "percentage"],
                "filters": [
                    {
                        "field": "diabetes_duration_years",
                        "operator": "gte",
                        "value": 5
                    }
                ]
            }
        }
    }

    result_multi_dim = {
        "buckets": [
            {
                "key": "<40",
                "from": 0,
                "to": 40,
                "doc_count": 300,
                "dimensions": {
                    "diabetic_retinopathy": {
                        "buckets": [
                            {"key": "Yes", "doc_count": 45, "percentage": 15.0},
                            {"key": "No", "doc_count": 255, "percentage": 85.0}
                        ]
                    },
                    "hypertension": {
                        "buckets": [
                            {"key": "Yes", "doc_count": 90, "percentage": 30.0},
                            {"key": "No", "doc_count": 210, "percentage": 70.0}
                        ]
                    }
                },
                "groups": {
                    "gender": {
                        "buckets": [
                            {"key": "Male", "doc_count": 165, "percentage": 55.0},
                            {"key": "Female", "doc_count": 135, "percentage": 45.0}
                        ]
                    }
                }
            },
            {
                "key": "40-60",
                "from": 40,
                "to": 60,
                "doc_count": 450,
                "dimensions": {
                    "diabetic_retinopathy": {
                        "buckets": [
                            {"key": "Yes", "doc_count": 135, "percentage": 30.0},
                            {"key": "No", "doc_count": 315, "percentage": 70.0}
                        ]
                    },
                    "hypertension": {
                        "buckets": [
                            {"key": "Yes", "doc_count": 225, "percentage": 50.0},
                            {"key": "No", "doc_count": 225, "percentage": 50.0}
                        ]
                    }
                },
                "groups": {
                    "gender": {
                        "buckets": [
                            {"key": "Male", "doc_count": 225, "percentage": 50.0},
                            {"key": "Female", "doc_count": 225, "percentage": 50.0}
                        ]
                    }
                }
            },
            {
                "key": "60+",
                "from": 60,
                "doc_count": 500,
                "dimensions": {
                    "diabetic_retinopathy": {
                        "buckets": [
                            {"key": "Yes", "doc_count": 250, "percentage": 50.0},
                            {"key": "No", "doc_count": 250, "percentage": 50.0}
                        ]
                    },
                    "hypertension": {
                        "buckets": [
                            {"key": "Yes", "doc_count": 350, "percentage": 70.0},
                            {"key": "No", "doc_count": 150, "percentage": 30.0}
                        ]
                    }
                },
                "groups": {
                    "gender": {
                        "buckets": [
                            {"key": "Male", "doc_count": 260, "percentage": 52.0},
                            {"key": "Female", "doc_count": 240, "percentage": 48.0}
                        ]
                    }
                }
            }
        ]
    }

    image_path3 = visualize_opsearch_results(query_multi_dim, result_multi_dim, "./examples")
    print(f"Multi-dimensional chart saved to: {image_path3}")

    # Example 4: Time series analysis (monthly disease trends)
    print("\nExample 4: Time series analysis - monthly disease trends")
    query_time_series = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["disease_type"],
                "buckets": [
                    {
                        "type": "date_histogram",
                        "field": "diagnosis_date",
                        "interval": "1M",
                        "format": "yyyy-MM"
                    }
                ],
                "metrics": ["count", "avg"],
                "metrics_field": "treatment_cost",
                "filters": [
                    {
                        "field": "diagnosis_date",
                        "operator": "range",
                        "value": {
                            "gte": "2023-01-01",
                            "lte": "2023-12-31"
                        }
                    }
                ]
            }
        }
    }

    result_time_series = {
        "buckets": [
            {
                "key": "2023-01",
                "doc_count": 120,
                "dimensions": {
                    "disease_type": {
                        "buckets": [
                            {"key": "Diabetes", "doc_count": 45, "metrics": {"avg": 1250.50}},
                            {"key": "Hypertension", "doc_count": 60, "metrics": {"avg": 850.75}},
                            {"key": "Asthma", "doc_count": 15, "metrics": {"avg": 320.25}}
                        ]
                    }
                },
                "metrics": {
                    "count": 120,
                    "avg": 950.25
                }
            },
            {
                "key": "2023-02",
                "doc_count": 135,
                "dimensions": {
                    "disease_type": {
                        "buckets": [
                            {"key": "Diabetes", "doc_count": 50, "metrics": {"avg": 1280.30}},
                            {"key": "Hypertension", "doc_count": 65, "metrics": {"avg": 870.40}},
                            {"key": "Asthma", "doc_count": 20, "metrics": {"avg": 315.80}}
                        ]
                    }
                },
                "metrics": {
                    "count": 135,
                    "avg": 965.45
                }
            },
            {
                "key": "2023-03",
                "doc_count": 140,
                "dimensions": {
                    "disease_type": {
                        "uckets": [
                            {"key": "Diabetes", "doc_count": 55, "metrics": {"avg": 1300.75}},
                            {"key": "Hypertension", "doc_count": 65, "metrics": {"avg": 880.20}},
                            {"key": "Asthma", "doc_count": 20, "metrics": {"avg": 325.60}}
                        ]
                    }
                },
                "metrics": {
                    "count": 140,
                    "avg": 985.30
                }
            }
        ]
    }

    image_path4 = visualize_opsearch_results(query_time_series, result_time_series, "./examples")
    print(f"Time series chart saved to: {image_path4}")

    # Example 5: Simple percentage distribution (disease prevalence)
    print("\nExample 5: Simple percentage distribution - disease prevalence")
    query_percentage = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["disease_status"],
                "metrics": ["percentage"],
                "filters": [
                    {
                        "field": "age",
                        "operator": "gte",
                        "value": 50
                    }
                ]
            }
        }
    }

    result_percentage = {
        "Diabetes": 25.5,
        "Hypertension": 42.3,
        "Asthma": 12.8,
        "COPD": 8.4,
        "Other": 11.0
    }

    image_path5 = visualize_opsearch_results(query_percentage, result_percentage, "./examples")
    print(f"Percentage distribution chart saved to: {image_path5}")

    # Example 6: Complex nested aggregation (treatment outcomes by multiple factors)
    print("\nExample 6: Complex nested aggregation - treatment outcomes")
    query_complex = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["treatment_outcome", "medication_type"],
                "groups": ["hospital", "doctor_specialty"],
                "buckets": [
                    {
                        "type": "range",
                        "field": "treatment_duration_days",
                        "ranges": [
                            {"key": "Short (<30 days)", "from": 0, "to": 30},
                            {"key": "Medium (30-90)", "from": 30, "to": 90},
                            {"key": "Long (90+)", "from": 90}
                        ]
                    },
                    {
                        "type": "terms",
                        "field": "patient_age_group",
                        "size": 5
                    }
                ],
                "metrics": ["count", "percentage", "avg", "max", "min"],
                "metrics_field": "treatment_cost",
                "filters": [
                    {
                        "field": "admission_date",
                        "operator": "gte",
                        "value": "2023-01-01"
                    }
                ]
            }
        }
    }

    result_complex = {
        "buckets": [
            {
                "key": "Short (<30 days)",
                "from": 0,
                "to": 30,
                "doc_count": 200,
                "dimensions": {
                    "treatment_outcome": {
                        "buckets": [
                            {
                                "key": "Recovered",
                                "doc_count": 150,
                                "dimensions": {
                                    "medication_type": {
                                        "buckets": [
                                            {"key": "Type A", "doc_count": 100,
                                             "metrics": {"avg": 1200, "max": 2500, "min": 500}},
                                            {"key": "Type B", "doc_count": 50,
                                             "metrics": {"avg": 1500, "max": 3000, "min": 600}}
                                        ]
                                    }
                                }
                            },
                            {
                                "key": "Improved",
                                "doc_count": 40,
                                "dimensions": {
                                    "medication_type": {
                                        "buckets": [
                                            {"key": "Type A", "doc_count": 25,
                                             "metrics": {"avg": 1800, "max": 3500, "min": 800}},
                                            {"key": "Type B", "doc_count": 15,
                                             "metrics": {"avg": 2200, "max": 4000, "min": 1000}}
                                        ]
                                    }
                                }
                            }
                        ]
                    }
                },
                "groups": {
                    "hospital": {
                        "buckets": [
                            {"key": "General Hospital", "doc_count": 120, "percentage": 60.0},
                            {"key": "Specialty Clinic", "doc_count": 80, "percentage": 40.0}
                        ]
                    }
                }
            }
        ]
    }

    image_path6 = visualize_opsearch_results(query_complex, result_complex, "./examples")
    print(f"Complex aggregation chart saved to: {image_path6}")

    print(f"\nAll charts saved to ./examples directory")
    print(f"Generated charts:")
    print(f"1. Statistical summary: {image_path1}")
    print(f"2. Gender percentage: {image_path2}")
    print(f"3. Multi-dimensional: {image_path3}")
    print(f"4. Time series: {image_path4}")
    print(f"5. Disease prevalence: {image_path5}")
    print(f"6. Complex aggregation: {image_path6}")