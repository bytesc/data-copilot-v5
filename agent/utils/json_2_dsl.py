from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import copy
import math


class OpenSearchQueryTranslator:
    """OpenSearch DSL查询翻译器"""

    def __init__(self, index_name: str = "_all"):
        self.index_name = index_name
        # 默认metrics配置
        self.default_metrics = {
            "descriptive_stats": ["count", "min", "max", "avg", "sum", "q1", "median", "q3"],
            "complete_stats": ["count", "min", "max", "avg", "sum", "q1", "q5", "median", "q3",
                               "std_deviation", "variance", "mode", "cardinality"],
            "frequency_analysis": ["count", "percentage"],
            "cross_analysis": ["count", "percentage"],
            "range_analysis": ["count"]
        }

    def translate(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        翻译JSON查询为OpenSearch DSL

        Args:
            query_json: JSON格式的查询定义

        Returns:
            OpenSearch DSL查询
        """
        query_type = query_json["query"]["type"]
        config = query_json["query"]["config"]

        # 根据查询类型路由到不同的处理方法
        if query_type == "descriptive_stats":
            return self._translate_descriptive_stats(config)
        elif query_type == "complete_stats":
            return self._translate_complete_stats(config)
        elif query_type == "frequency_analysis":
            return self._translate_frequency_analysis(config)
        elif query_type == "cross_analysis":
            return self._translate_cross_analysis(config)
        elif query_type == "range_analysis":
            return self._translate_range_analysis(config)
        else:
            raise ValueError(f"不支持的查询类型: {query_type}")

    def _build_base_query(self, filters: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """构建基础查询"""
        query = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": [],
                    "must_not": [],
                    "should": []
                }
            }
        }

        if filters:
            for filter_cond in filters:
                self._add_filter(query, filter_cond)

        return query

    def _add_filter(self, query: Dict[str, Any], filter_cond: Dict[str, Any]):
        """添加过滤条件"""
        field = filter_cond["field"]
        operator = filter_cond["operator"]
        value = filter_cond.get("value")

        if operator == "eq":
            term_filter = {"term": {field: value}}
            query["query"]["bool"]["filter"].append(term_filter)

        elif operator in ["gt", "gte", "lt", "lte"]:
            range_filter = {"range": {field: {operator: value}}}
            query["query"]["bool"]["filter"].append(range_filter)

        elif operator == "in":
            terms_filter = {"terms": {field: value}}
            query["query"]["bool"]["filter"].append(terms_filter)

        elif operator == "neq":
            term_filter = {"term": {field: value}}
            query["query"]["bool"]["must_not"].append(term_filter)

        elif operator == "range":
            range_filter = {"range": {field: value}}
            query["query"]["bool"]["filter"].append(range_filter)

        elif operator == "like":
            # SQL LIKE模式转换为wildcard
            wildcard_value = value.replace('%', '*').replace('_', '?')
            wildcard_filter = {"wildcard": {field: wildcard_value}}
            query["query"]["bool"]["filter"].append(wildcard_filter)

        elif operator == "wildcard":
            wildcard_filter = {"wildcard": {field: value}}
            query["query"]["bool"]["filter"].append(wildcard_filter)

        elif operator == "regexp":
            regexp_filter = {"regexp": {field: value}}
            query["query"]["bool"]["filter"].append(regexp_filter)

        elif operator == "exists":
            exists_filter = {"exists": {"field": field}}
            query["query"]["bool"]["filter"].append(exists_filter)

        elif operator == "missing":
            exists_filter = {"exists": {"field": field}}
            query["query"]["bool"]["must_not"].append(exists_filter)

    def _get_metrics_aggregations(self, field: str, metrics: List[str],
                                  metrics_field: Optional[str] = None) -> Dict[str, Any]:
        """根据metrics列表构建聚合"""
        aggs = {}
        target_field = metrics_field or field

        for metric in metrics:
            if metric == "count":
                # count是bucket的doc_count，不需要单独聚合
                continue
            elif metric == "cardinality":
                aggs[f"{field}_cardinality"] = {
                    "cardinality": {"field": target_field}
                }
            elif metric == "min":
                aggs[f"{field}_min"] = {
                    "min": {"field": target_field}
                }
            elif metric == "max":
                aggs[f"{field}_max"] = {
                    "max": {"field": target_field}
                }
            elif metric == "avg":
                aggs[f"{field}_avg"] = {
                    "avg": {"field": target_field}
                }
            elif metric == "sum":
                aggs[f"{field}_sum"] = {
                    "sum": {"field": target_field}
                }
            elif metric == "median":
                aggs[f"{field}_percentiles"] = {
                    "percentiles": {
                        "field": target_field,
                        "percents": [50]
                    }
                }
            elif metric == "q1":
                if f"{field}_percentiles" not in aggs:
                    aggs[f"{field}_percentiles"] = {
                        "percentiles": {
                            "field": target_field,
                            "percents": [25, 50, 75]
                        }
                    }
            elif metric == "q3":
                if f"{field}_percentiles" not in aggs:
                    aggs[f"{field}_percentiles"] = {
                        "percentiles": {
                            "field": target_field,
                            "percents": [25, 50, 75]
                        }
                    }
            elif metric == "q5":
                if f"{field}_q5" not in aggs:
                    aggs[f"{field}_q5"] = {
                        "percentiles": {
                            "field": target_field,
                            "percents": [5]
                        }
                    }
            elif metric == "std_deviation":
                aggs[f"{field}_extended_stats"] = {
                    "extended_stats": {"field": target_field}
                }
            elif metric == "variance":
                if f"{field}_extended_stats" not in aggs:
                    aggs[f"{field}_extended_stats"] = {
                        "extended_stats": {"field": target_field}
                    }
            elif metric == "mode":
                aggs[f"{field}_terms"] = {
                    "terms": {"field": target_field, "size": 1}
                }

        return aggs

    def _translate_descriptive_stats(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译描述性统计查询"""
        fields = config.get("fields", [])
        filters = config.get("filters", [])
        metrics = config.get("metrics", self.default_metrics["descriptive_stats"])

        if not fields:
            raise ValueError("descriptive_stats查询必须指定fields参数")

        # 构建基础查询
        query = self._build_base_query(filters)

        # 添加聚合统计
        aggs = {}
        for field in fields:
            field_aggs = self._get_metrics_aggregations(field, metrics)
            aggs.update(field_aggs)

        query["aggs"] = aggs
        query["size"] = 0  # 不返回原始数据

        return query

    def _translate_complete_stats(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译完整统计查询"""
        fields = config.get("fields", [])
        filters = config.get("filters", [])
        metrics = config.get("metrics", self.default_metrics["complete_stats"])

        if not fields:
            raise ValueError("complete_stats查询必须指定fields参数")

        # 构建基础查询
        query = self._build_base_query(filters)

        # 添加完整的聚合统计
        aggs = {}
        for field in fields:
            field_aggs = self._get_metrics_aggregations(field, metrics)
            aggs.update(field_aggs)

        query["aggs"] = aggs
        query["size"] = 0

        return query

    def _translate_frequency_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译频率分析查询"""
        fields = config.get("fields", [])
        group_by = config.get("group_by", [])
        filters = config.get("filters", [])
        metrics = config.get("metrics", self.default_metrics["frequency_analysis"])

        if not fields:
            raise ValueError("frequency_analysis查询必须指定fields参数")

        # 构建基础查询
        query = self._build_base_query(filters)
        query["size"] = 0

        # 构建聚合层级
        aggs = self._build_nested_aggregations(fields, group_by, metrics, None)
        query["aggs"] = aggs

        return query

    def _build_nested_aggregations(self, fields: List[str], group_by: List[str],
                                   metrics: List[str], bucket_ranges: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """构建嵌套聚合结构"""
        aggs = {}
        current_agg = aggs

        # 如果有bucket_ranges，先构建范围聚合
        if bucket_ranges:
            for i, bucket_range in enumerate(bucket_ranges):
                range_field = bucket_range["field"]
                ranges = bucket_range["ranges"]

                range_aggs = {
                    "range": {
                        "field": range_field,
                        "ranges": []
                    },
                    "aggs": {}
                }

                for r in ranges:
                    range_def = {"key": r["key"]}
                    if "from" in r:
                        range_def["from"] = r["from"]
                    if "to" in r:
                        range_def["to"] = r["to"]
                    range_aggs["range"]["ranges"].append(range_def)

                current_agg[range_field] = range_aggs
                current_agg = current_agg[range_field]["aggs"]

        # 构建分组字段聚合
        for i, group_field in enumerate(group_by):
            current_agg[group_field] = {
                "terms": {
                    "field": group_field,
                    "size": 100
                },
                "aggs": {}
            }
            current_agg = current_agg[group_field]["aggs"]

        # 添加字段统计
        for field in fields:
            # 添加基数统计
            if "cardinality" in metrics:
                current_agg[f"{field}_cardinality"] = {
                    "cardinality": {"field": field}
                }

            # 添加术语聚合
            current_agg[f"{field}_terms"] = {
                "terms": {"field": field, "size": 100}
            }

            # 如果需要百分比，添加bucket_script
            if "percentage" in metrics:
                if len(group_by) > 0 or bucket_ranges:
                    # 有分组时，计算相对于父级的百分比
                    current_agg[f"{field}_terms"]["aggs"] = {
                        "percentage": {
                            "bucket_script": {
                                "buckets_path": {
                                    "count": "_count",
                                    "total": "_parent._count"
                                },
                                "script": "params.count / params.total * 100"
                            }
                        }
                    }
                else:
                    # 无分组时，计算相对于总数的百分比
                    current_agg[f"{field}_terms"]["aggs"] = {
                        "percentage": {
                            "bucket_script": {
                                "buckets_path": {
                                    "count": "_count",
                                    "total": "_count"
                                },
                                "script": "params.count / params.total * 100"
                            }
                        }
                    }

        return aggs

    def _translate_cross_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译交叉分析查询"""
        fields = config.get("fields", [])
        group_by = config.get("group_by", [])
        bucket_ranges = config.get("bucket_ranges", [])
        filters = config.get("filters", [])
        metrics = config.get("metrics", self.default_metrics["cross_analysis"])

        if not fields:
            raise ValueError("cross_analysis查询必须指定fields参数")

        # 构建基础查询
        query = self._build_base_query(filters)
        query["size"] = 0

        # 构建聚合结构
        aggs = self._build_nested_aggregations(fields, group_by, metrics, bucket_ranges)
        query["aggs"] = aggs

        return query

    def _translate_range_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译范围分析查询"""
        target_field = config.get("field")
        ranges = config.get("ranges", [])
        group_by = config.get("group_by", [])
        metrics_field = config.get("metrics_field")
        filters = config.get("filters", [])
        metrics = config.get("metrics", self.default_metrics["range_analysis"])

        if not target_field or not ranges:
            raise ValueError("range_analysis查询必须指定field和ranges参数")

        # 构建基础查询
        query = self._build_base_query(filters)
        query["size"] = 0

        # 构建范围聚合
        aggs = {}
        current_agg = aggs

        # 添加范围聚合
        current_agg["ranges"] = {
            "range": {
                "field": target_field,
                "ranges": []
            },
            "aggs": {}
        }

        # 添加范围定义
        for r in ranges:
            range_def = {"key": r["key"]}
            if "from" in r:
                range_def["from"] = r["from"]
            if "to" in r:
                range_def["to"] = r["to"]
            current_agg["ranges"]["range"]["ranges"].append(range_def)

        current_agg = current_agg["ranges"]["aggs"]

        # 如果有分组字段，添加嵌套分组
        for i, group_field in enumerate(group_by):
            current_agg[group_field] = {
                "terms": {
                    "field": group_field,
                    "size": 50
                },
                "aggs": {}
            }
            current_agg = current_agg[group_field]["aggs"]

        # 添加metrics统计
        if metrics_field and metrics:
            metrics_to_calc = [m for m in metrics if m not in ["count", "percentage"]]
            if metrics_to_calc:
                field_aggs = self._get_metrics_aggregations(metrics_field, metrics_to_calc, metrics_field)
                for agg_name, agg_def in field_aggs.items():
                    # 修改聚合名称，避免冲突
                    new_agg_name = f"range_{agg_name}"
                    current_agg[new_agg_name] = agg_def

        query["aggs"] = aggs
        return query

    def translate_with_pagination(self, query_json: Dict[str, Any],
                                  page: int = 1,
                                  size: int = 10) -> Dict[str, Any]:
        """翻译查询并添加分页"""
        query = self.translate(query_json)

        # 添加分页
        query["from"] = (page - 1) * size
        query["size"] = size

        return query

    def format_result(self, query_type: str, es_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化OpenSearch返回结果"""
        if query_type in ["descriptive_stats", "complete_stats"]:
            return self._format_stats_result(es_result, query_type)
        elif query_type in ["frequency_analysis", "cross_analysis"]:
            return self._format_aggregation_result(es_result)
        elif query_type == "range_analysis":
            return self._format_range_result(es_result)
        else:
            return es_result

    def _format_stats_result(self, es_result: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """格式化统计结果"""
        formatted = {}
        aggs = es_result.get("aggregations", {})

        for agg_name, agg_data in aggs.items():
            # 解析字段名
            if agg_name.endswith("_stats"):
                field = agg_name.replace("_stats", "")
                if field not in formatted:
                    formatted[field] = {}
                formatted[field].update({
                    "min": agg_data.get("min"),
                    "max": agg_data.get("max"),
                    "avg": agg_data.get("avg"),
                    "sum": agg_data.get("sum"),
                    "count": agg_data.get("count")
                })
            elif agg_name.endswith("_extended_stats"):
                field = agg_name.replace("_extended_stats", "")
                if field not in formatted:
                    formatted[field] = {}
                formatted[field].update({
                    "std_deviation": agg_data.get("std_deviation"),
                    "variance": agg_data.get("variance"),
                    "min": agg_data.get("min"),
                    "max": agg_data.get("max"),
                    "avg": agg_data.get("avg"),
                    "sum": agg_data.get("sum"),
                    "count": agg_data.get("count")
                })
            elif agg_name.endswith("_percentiles"):
                field = agg_name.replace("_percentiles", "")
                if field not in formatted:
                    formatted[field] = {}
                values = agg_data.get("values", {})
                if "25.0" in values:
                    formatted[field]["q1"] = values["25.0"]
                if "50.0" in values:
                    formatted[field]["median"] = values["50.0"]
                if "75.0" in values:
                    formatted[field]["q3"] = values["75.0"]
            elif agg_name.endswith("_q5"):
                field = agg_name.replace("_q5", "")
                if field not in formatted:
                    formatted[field] = {}
                values = agg_data.get("values", {})
                if "5.0" in values:
                    formatted[field]["q5"] = values["5.0"]
            elif agg_name.endswith("_cardinality"):
                field = agg_name.replace("_cardinality", "")
                if field not in formatted:
                    formatted[field] = {}
                formatted[field]["cardinality"] = agg_data.get("value")
            elif agg_name.endswith("_terms"):
                field = agg_name.replace("_terms", "")
                if field not in formatted:
                    formatted[field] = {}
                buckets = agg_data.get("buckets", [])
                if buckets:
                    formatted[field]["mode"] = buckets[0].get("key")
                    formatted[field]["mode_count"] = buckets[0].get("doc_count")
            elif agg_name in ["_min", "_max", "_avg", "_sum"]:
                field = agg_name[1:]  # 去掉下划线前缀
                if field not in formatted:
                    formatted[field] = {}
                formatted[field][agg_name[1:]] = agg_data.get("value")

        return formatted

    def _format_aggregation_result(self, es_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化聚合结果"""

        def process_buckets(buckets, level=0, parent_key=None, parent_path=""):
            results = []
            for bucket in buckets.get("buckets", []):
                result = {
                    "key": bucket.get("key"),
                    "doc_count": bucket.get("doc_count"),
                    "level": level
                }
                if parent_key is not None:
                    result["parent_key"] = parent_key
                if parent_path:
                    result["path"] = f"{parent_path}.{bucket.get('key')}"

                # 处理统计信息
                for key, value in bucket.items():
                    if key not in ["key", "doc_count", "key_as_string"]:
                        if isinstance(value, dict):
                            if "buckets" in value:
                                # 嵌套聚合
                                result["children"] = process_buckets(
                                    value, level + 1, bucket.get("key"),
                                    f"{parent_path}.{bucket.get('key')}" if parent_path else str(bucket.get("key"))
                                )
                            elif "value" in value:
                                # 聚合值
                                result[key] = value["value"]
                            elif "values" in value:
                                # 百分比聚合
                                result[key] = value["values"]
                            elif "ranges" in value:
                                # 范围聚合
                                result["ranges"] = []
                                for range_bucket in value.get("buckets", []):
                                    range_result = {
                                        "key": range_bucket.get("key"),
                                        "from": range_bucket.get("from"),
                                        "to": range_bucket.get("to"),
                                        "doc_count": range_bucket.get("doc_count")
                                    }
                                    # 处理范围内的子聚合
                                    for sub_key, sub_value in range_bucket.items():
                                        if sub_key not in ["key", "from", "to", "doc_count", "key_as_string"]:
                                            if "buckets" in sub_value:
                                                range_result["children"] = process_buckets(
                                                    sub_value, level + 1,
                                                    f"{bucket.get('key')}:{range_bucket.get('key')}",
                                                    f"{parent_path}.{bucket.get('key')}.{range_bucket.get('key')}"
                                                    if parent_path else f"{bucket.get('key')}.{range_bucket.get('key')}"
                                                )
                                    result["ranges"].append(range_result)

                # 处理percentage计算
                if "percentage" in bucket:
                    result["percentage"] = bucket["percentage"]["value"]

                results.append(result)
            return results

        aggs = es_result.get("aggregations", {})
        return {"buckets": process_buckets(aggs)}

    def _format_range_result(self, es_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化范围分析结果"""
        formatted = {"ranges": []}
        aggs = es_result.get("aggregations", {})

        if "ranges" in aggs:
            for bucket in aggs["ranges"].get("buckets", []):
                range_data = {
                    "key": bucket.get("key"),
                    "from": bucket.get("from"),
                    "to": bucket.get("to"),
                    "doc_count": bucket.get("doc_count")
                }

                # 处理分组
                groups = []
                for key, value in bucket.items():
                    if key not in ["key", "from", "to", "doc_count", "key_as_string"]:
                        if "buckets" in value:
                            for group_bucket in value.get("buckets", []):
                                group_data = {
                                    "key": group_bucket.get("key"),
                                    "doc_count": group_bucket.get("doc_count")
                                }
                                # 添加统计信息
                                for sub_key, sub_value in group_bucket.items():
                                    if sub_key not in ["key", "doc_count", "key_as_string"]:
                                        if "value" in sub_value:
                                            # 提取metric名称
                                            metric_name = sub_key.replace("range_", "")
                                            group_data[metric_name] = sub_value.get("value")
                                groups.append(group_data)
                        elif "value" in value:
                            # 直接统计值
                            metric_name = key.replace("range_", "")
                            range_data[metric_name] = value.get("value")

                if groups:
                    range_data["groups"] = groups

                formatted["ranges"].append(range_data)

        return formatted


# 使用示例
if __name__ == "__main__":
    # 创建翻译器实例
    translator = OpenSearchQueryTranslator(index_name="medical_records")

    # 示例1：描述性统计
    stats_query = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "fields": ["age", "blood_pressure", "cholesterol"],
                "filters": [
                    {
                        "field": "gender",
                        "operator": "eq",
                        "value": "male"
                    },
                    {
                        "field": "age",
                        "operator": "gte",
                        "value": 30
                    }
                ]
            }
        }
    }

    # 示例2：完整统计
    complete_stats_query = {
        "query": {
            "type": "complete_stats",
            "config": {
                "fields": ["age", "blood_pressure"],
                "metrics": ["min", "max", "avg", "median", "q1", "q5", "q3", "std_deviation", "mode"],
                "filters": [
                    {
                        "field": "gender",
                        "operator": "eq",
                        "value": "female"
                    }
                ]
            }
        }
    }

    # 示例3：频率分析
    freq_query = {
        "query": {
            "type": "frequency_analysis",
            "config": {
                "fields": ["education_level", "job_title"],
                "group_by": ["department"],
                "metrics": ["count", "percentage"],
                "filters": [
                    {
                        "field": "active",
                        "operator": "eq",
                        "value": True
                    }
                ]
            }
        }
    }

    # 示例4：交叉分析（带自定义范围）
    cross_query = {
        "query": {
            "type": "cross_analysis",
            "config": {
                "fields": ["has_disease", "treatment_type"],
                "group_by": ["gender"],
                "bucket_ranges": [
                    {
                        "field": "age",
                        "ranges": [
                            {"key": "young", "from": 0, "to": 30},
                            {"key": "middle", "from": 30, "to": 60},
                            {"key": "old", "from": 60}
                        ],
                        "type": "range"
                    }
                ],
                "metrics": ["count", "percentage"],
                "filters": [
                    {
                        "field": "test_date",
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

    # 示例5：范围分析
    range_query = {
        "query": {
            "type": "range_analysis",
            "config": {
                "field": "age",
                "ranges": [
                    {"key": "0-20", "from": 0, "to": 20},
                    {"key": "20-40", "from": 20, "to": 40},
                    {"key": "40-60", "from": 40, "to": 60},
                    {"key": "60-80", "from": 60, "to": 80},
                    {"key": "80+", "from": 80}
                ],
                "group_by": ["has_disease", "disease_type"],
                "metrics_field": "age",
                "metrics": ["count", "avg"],
                "filters": [
                    {
                        "field": "test_date",
                        "operator": "gte",
                        "value": "2023-01-01"
                    }
                ]
            }
        }
    }

    # 示例6：使用exists操作符
    exists_query = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "fields": ["age"],
                "filters": [
                    {
                        "field": "email",
                        "operator": "exists"
                    },
                    {
                        "field": "phone",
                        "operator": "missing"
                    }
                ]
            }
        }
    }

    # 示例7：使用正则表达式
    regex_query = {
        "query": {
            "type": "frequency_analysis",
            "config": {
                "fields": ["name"],
                "filters": [
                    {
                        "field": "name",
                        "operator": "regexp",
                        "value": "张.*"
                    }
                ]
            }
        }
    }

    # 测试所有查询
    import json

    test_queries = [
        ("描述性统计", stats_query),
        ("完整统计", complete_stats_query),
        ("频率分析", freq_query),
        ("交叉分析", cross_query),
        ("范围分析", range_query),
        ("exists查询", exists_query),
        ("正则查询", regex_query)
    ]

    for query_name, query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"查询类型: {query_name}")
        print(f"{'=' * 60}")
        try:
            es_query = translator.translate(query)
            print("生成的OpenSearch查询:")
            print(json.dumps(es_query, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"错误: {e}")

    # 模拟OpenSearch返回结果
    mock_stats_result = {
        "aggregations": {
            "age_extended_stats": {
                "count": 100,
                "min": 25.0,
                "max": 80.0,
                "avg": 52.5,
                "sum": 5250.0,
                "std_deviation": 12.5,
                "variance": 156.25
            },
            "age_percentiles": {
                "values": {
                    "25.0": 40.0,
                    "50.0": 52.5,
                    "75.0": 65.0
                }
            },
            "age_q5": {
                "values": {
                    "5.0": 30.0
                }
            },
            "age_cardinality": {
                "value": 56
            },
            "age_terms": {
                "buckets": [
                    {"key": 35, "doc_count": 15}
                ]
            }
        }
    }

    # 测试结果格式化
    print("\n\n结果格式化测试:")
    formatted = translator.format_result("complete_stats", mock_stats_result)
    print(json.dumps(formatted, indent=2, ensure_ascii=False))