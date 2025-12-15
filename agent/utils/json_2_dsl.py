from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import copy

from agent.utils.search_tools.opensearch_connection import search_by_dsl


class OpenSearchQueryTranslator:
    """OpenSearch DSL查询翻译器"""

    def __init__(self, index_name: str = "_all"):
        self.index_name = index_name
        self.operators_map = {
            "eq": "term",
            "neq": "must_not",
            "gt": "gt",
            "gte": "gte",
            "lt": "lt",
            "lte": "lte",
            "in": "terms",
            "like": "wildcard",
            "range": "range"
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
        elif query_type == "frequency_analysis":
            return self._translate_frequency_analysis(config)
        elif query_type == "cross_analysis":
            return self._translate_cross_analysis(config)
        else:
            raise ValueError(f"不支持的查询类型: {query_type}")

    def _build_base_query(self, filters: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """构建基础查询"""
        query = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": [],
                    "must_not": []
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

    def _translate_descriptive_stats(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译描述性统计查询"""
        fields = config.get("fields", [])
        filters = config.get("filters", [])

        # 构建基础查询
        query = self._build_base_query(filters)

        # 添加聚合统计
        aggs = {}
        for field in fields:
            # 数值字段的统计聚合
            aggs.update({
                f"{field}_stats": {
                    "stats": {"field": field}
                },
                f"{field}_percentiles": {
                    "percentiles": {
                        "field": field,
                        "percents": [25, 50, 75]
                    }
                }
            })

        # 添加基数统计（唯一值数量）
        for field in fields:
            aggs[f"{field}_cardinality"] = {
                "cardinality": {"field": field}
            }

        query["aggs"] = aggs
        query["size"] = 0  # 不返回原始数据

        return query

    def _translate_frequency_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译频率分析查询"""
        fields = config.get("fields", [])
        group_by = config.get("group_by", [])
        filters = config.get("filters", [])

        # 构建基础查询
        query = self._build_base_query(filters)
        query["size"] = 0

        # 构建聚合层级
        aggs = {}
        current_agg = aggs

        # 如果有分组字段，创建嵌套聚合
        for i, group_field in enumerate(group_by):
            current_agg[group_field] = {
                "terms": {
                    "field": group_field,
                    "size": 100  # 可调整大小
                },
                "aggs": {}
            }
            if i < len(group_by) - 1:
                current_agg = current_agg[group_field]["aggs"]
            else:
                # 最后一层添加字段统计
                for field in fields:
                    current_agg[group_field]["aggs"][f"{field}_terms"] = {
                        "terms": {"field": field, "size": 100}
                    }
        else:
            # 没有分组字段，直接统计
            for field in fields:
                aggs[f"{field}_terms"] = {
                    "terms": {"field": field, "size": 100}
                }

        query["aggs"] = aggs
        return query

    def _translate_cross_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """翻译交叉分析查询（如疾病比例）"""
        fields = config.get("fields", [])
        group_by = config.get("group_by", [])
        metrics = config.get("metrics", ["count", "percentage"])
        filters = config.get("filters", [])

        # 构建基础查询
        query = self._build_base_query(filters)
        query["size"] = 0

        aggs = {}
        current_agg = aggs

        # 构建分组聚合
        for i, group_field in enumerate(group_by):
            current_agg[group_field] = {
                "terms": {
                    "field": group_field,
                    "size": 50
                },
                "aggs": {}
            }
            current_agg = current_agg[group_field]["aggs"]

        # 为每个字段添加统计
        for field in fields:
            # 添加基数统计
            current_agg[f"{field}_cardinality"] = {
                "cardinality": {"field": field}
            }

            # 添加术语聚合
            current_agg[f"{field}_terms"] = {
                "terms": {"field": field, "size": 10}
            }

            # 如果需要百分比，添加bucket_script
            if "percentage" in metrics:
                current_agg[f"{field}_terms"]["aggs"] = {
                    "percentage": {
                        "bucket_script": {
                            "buckets_path": {
                                "count": "_count",
                                "total": f"_parent._count"
                            },
                            "script": "params.count / params.total * 100"
                        }
                    }
                }

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
        if query_type == "descriptive_stats":
            return self._format_descriptive_result(es_result)
        elif query_type in ["frequency_analysis", "cross_analysis"]:
            return self._format_aggregation_result(es_result)
        else:
            return es_result

    def _format_descriptive_result(self, es_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化描述性统计结果"""
        formatted = {}
        aggs = es_result.get("aggregations", {})

        for agg_name, agg_data in aggs.items():
            if agg_name.endswith("_stats"):
                field = agg_name.replace("_stats", "")
                formatted[field] = {
                    "min": agg_data.get("min"),
                    "max": agg_data.get("max"),
                    "avg": agg_data.get("avg"),
                    "sum": agg_data.get("sum"),
                    "count": agg_data.get("count")
                }
            elif agg_name.endswith("_percentiles"):
                field = agg_name.replace("_percentiles", "")
                if field not in formatted:
                    formatted[field] = {}
                formatted[field].update({
                    "q1": agg_data.get("values", {}).get("25.0"),
                    "median": agg_data.get("values", {}).get("50.0"),
                    "q3": agg_data.get("values", {}).get("75.0")
                })

        return formatted

    def _format_aggregation_result(self, es_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """格式化聚合结果"""

        def process_buckets(buckets, level=0, parent_key=None):
            results = []
            for bucket in buckets.get("buckets", []):
                result = {
                    "key": bucket.get("key"),
                    "doc_count": bucket.get("doc_count"),
                    "level": level
                }
                if parent_key:
                    result["parent_key"] = parent_key

                # 递归处理子聚合
                for key, value in bucket.items():
                    if key not in ["key", "doc_count", "key_as_string"]:
                        if "buckets" in value:
                            result["children"] = process_buckets(
                                value, level + 1, bucket.get("key")
                            )
                        elif isinstance(value, dict) and "value" in value:
                            result[key] = value["value"]

                results.append(result)
            return results

        aggs = es_result.get("aggregations", {})
        return process_buckets(aggs)


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

    # 示例2：疾病比例分析
    disease_query = {
        "query": {
            "type": "cross_analysis",
            "config": {
                "fields": ["has_disease"],
                "group_by": ["age_group"],
                "metrics": ["count", "percentage"],
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

    # 翻译查询
    es_query1 = translator.translate(stats_query)
    es_query2 = translator.translate(disease_query)

    print("示例1 - 描述性统计查询:")
    import json

    print(json.dumps(es_query1, indent=2, ensure_ascii=False))

    print("\n示例2 - 疾病比例分析查询:")
    print(json.dumps(es_query2, indent=2, ensure_ascii=False))


    search_by_dsl()


