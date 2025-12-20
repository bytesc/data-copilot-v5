import json
from typing import Dict, List, Any, Optional
from collections import defaultdict


class OpenSearchStatsTranslator:
    """OpenSearch统计分析JSON翻译器"""

    # 支持的统计指标映射
    STATS_METRICS_MAP = {
        'count': 'value_count',
        'min': 'min',
        'max': 'max',
        'avg': 'avg',
        'sum': 'sum',
        'median': 'percentiles',
        'q1': 'percentiles',
        'q3': 'percentiles',
        'q5': 'percentiles',
        'std_deviation': 'extended_stats',
        'variance': 'extended_stats',
        'mode': 'terms',
        'cardinality': 'cardinality'
    }

    def __init__(self):
        self.query = {}

    def translate(self, input_json: Dict) -> Dict:
        """主翻译方法：JSON配置转OpenSearch DSL"""
        try:
            query_type = input_json['query']['type']
            config = input_json['query']['config']

            self._build_base_query(config.get('filters', []))

            if query_type == 'stats':
                return self._build_stats_query(config)
            elif query_type == 'distribution':
                return self._build_distribution_query(config)
            else:
                raise ValueError(f"不支持的查询类型: {query_type}")

        except Exception as e:
            return {'error': str(e)}

    def _build_base_query(self, filters: List[Dict]) -> None:
        """构建基础查询条件"""
        if not filters:
            self.query = {'match_all': {}}
            return

        bool_query = {'bool': {'must': []}}

        for filter_cond in filters:
            field = filter_cond['field']
            operator = filter_cond['operator']
            value = filter_cond.get('value')

            condition = self._build_filter_condition(field, operator, value)
            if condition:
                bool_query['bool']['must'].append(condition)

        self.query = bool_query

    def _build_filter_condition(self, field: str, operator: str, value: Any) -> Optional[Dict]:
        """构建单个过滤条件"""
        if operator == 'eq':
            return {'term': {field: value}}
        elif operator == 'neq':
            return {'bool': {'must_not': [{'term': {field: value}}]}}
        elif operator == 'gt':
            return {'range': {field: {'gt': value}}}
        elif operator == 'gte':
            return {'range': {field: {'gte': value}}}
        elif operator == 'lt':
            return {'range': {field: {'lt': value}}}
        elif operator == 'lte':
            return {'range': {field: {'lte': value}}}
        elif operator == 'in':
            return {'terms': {field: value}}
        elif operator == 'range':
            return {'range': {field: value}}
        elif operator == 'exists':
            return {'exists': {'field': field}}
        elif operator == 'missing':
            return {'bool': {'must_not': [{'exists': {'field': field}}]}}
        else:
            return None

    def _build_stats_query(self, config: Dict) -> Dict:
        """构建统计计算查询"""
        fields = config.get('fields', [])
        metrics = config.get('metrics', ['min', 'max', 'avg', 'count', 'q1', 'median', 'q3'])

        aggs = {}
        for field in fields:
            field_aggs = {}
            for metric in metrics:
                if metric in ['median', 'q1', 'q3', 'q5']:
                    if 'percentiles' not in field_aggs:
                        field_aggs['percentiles'] = {
                            'percentiles': {'field': field, 'percents': []}
                        }
                    percent_value = 50 if metric == 'median' else 25 if metric == 'q1' else 75 if metric == 'q3' else 5
                    if percent_value not in field_aggs['percentiles']['percentiles']['percents']:
                        field_aggs['percentiles']['percentiles']['percents'].append(percent_value)

                elif metric in ['std_deviation', 'variance']:
                    if 'extended_stats' not in field_aggs:
                        field_aggs['extended_stats'] = {'extended_stats': {'field': field}}

                elif metric == 'mode':
                    field_aggs['mode'] = {
                        'terms': {'field': field, 'size': 1}
                    }

                else:
                    es_metric = self.STATS_METRICS_MAP.get(metric, metric)
                    field_aggs[metric] = {es_metric: {'field': field}}

            aggs[field] = {'aggs': field_aggs}

        return {
            'size': 0,
            'query': self.query,
            'aggs': aggs
        }

    def _build_distribution_query(self, config: Dict) -> Dict:
        """构建分布分析查询 ，支持百分比计算"""
        dimensions = config.get('dimensions', [])
        groups = config.get('groups', [])
        buckets = config.get('buckets', [])
        metrics = config.get('metrics', ['count', 'percentage'])
        metrics_field = config.get('metrics_field')

        # 构建聚合结构
        aggs = self._build_distribution_aggregations(
            dimensions, groups, buckets, metrics, metrics_field
        )

        return {
            'size': 0,
            'query': self.query,
            'aggs': aggs
        }

    def _build_distribution_aggregations(self, dimensions: List[str], groups: List[str],
                                         buckets: List[Dict], metrics: List[str],
                                         metrics_field: str) -> Dict:
        """构建分布分析的聚合结构"""
        aggs = {}
        current_level = aggs

        # 添加总计数用于百分比计算
        if 'percentage' in metrics:
            current_level['_total_count'] = {'value_count': {'field': '_index'}}

        # 构建分组层级
        for group_field in groups:
            current_level[group_field] = {
                'terms': {'field': group_field, 'size': 100},
                'aggs': {
                    '_group_count': {'value_count': {'field': '_index'}}  # 分组级别计数
                }
            }
            current_level = current_level[group_field]['aggs']

        # 构建桶聚合
        for bucket in buckets:
            bucket_type = bucket['type']
            bucket_field = bucket['field']

            if bucket_type == 'terms':
                current_level[bucket_field] = {
                    'terms': {'field': bucket_field, 'size': bucket.get('size', 10)},
                    'aggs': {
                        '_bucket_count': {'value_count': {'field': '_index'}}  # 桶级别计数
                    }
                }
                current_level = current_level[bucket_field]['aggs']

            elif bucket_type == 'range':
                range_ranges = []
                for range_def in bucket['ranges']:
                    range_spec = {}
                    if 'from' in range_def:
                        range_spec['from'] = range_def['from']
                    if 'to' in range_def:
                        range_spec['to'] = range_def['to']
                    if 'key' in range_def:
                        range_spec['key'] = range_def['key']
                    range_ranges.append(range_spec)

                current_level[bucket_field] = {
                    'range': {'field': bucket_field, 'ranges': range_ranges},
                    'aggs': {
                        '_bucket_count': {'value_count': {'field': '_index'}}
                    }
                }
                current_level = current_level[bucket_field]['aggs']

            elif bucket_type == 'date_histogram':
                current_level[bucket_field] = {
                    'date_histogram': {
                        'field': bucket_field,
                        'interval': bucket['interval'],
                        'format': bucket.get('format', 'yyyy-MM')
                    },
                    'aggs': {
                        '_bucket_count': {'value_count': {'field': '_index'}}
                    }
                }
                current_level = current_level[bucket_field]['aggs']

        # 构建维度聚合
        for dim_field in dimensions:
            current_level[dim_field] = {
                'terms': {'field': dim_field, 'size': 100},
                'aggs': {
                    '_dimension_count': {'value_count': {'field': '_index'}}
                }
            }
            current_level = current_level[dim_field]['aggs']

        # 添加指标计算
        self._add_metrics_aggregations(current_level, metrics, metrics_field)

        return aggs

    def _add_metrics_aggregations(self, aggs: Dict, metrics: List[str], metrics_field: str) -> None:
        """添加指标计算聚合"""
        for metric in metrics:
            if metric == 'count':
                aggs['count'] = {'value_count': {'field': '_index'}}
            elif metric in ['avg', 'sum', 'min', 'max'] and metrics_field:
                es_metric = self.STATS_METRICS_MAP.get(metric, metric)
                aggs[metric] = {es_metric: {'field': metrics_field}}

    def process_stats_result(self, es_result: Dict, original_config: Dict) -> Dict:
        """处理统计计算结果"""
        try:
            result = {}
            config = original_config['query']['config']
            fields = config.get('fields', [])
            metrics = config.get('metrics', ['min', 'max', 'avg', 'count', 'q1', 'median', 'q3'])

            aggregations = es_result.get('aggregations', {})

            for field in fields:
                field_result = {}
                field_aggs = aggregations.get(field, {})

                for metric in metrics:
                    if metric == 'count':
                        field_result['count'] = field_aggs.get('count', {}).get('value', 0)
                    elif metric in ['min', 'max', 'avg', 'sum']:
                        field_result[metric] = field_aggs.get(metric, {}).get('value')
                    elif metric in ['std_deviation', 'variance']:
                        ext_stats = field_aggs.get('extended_stats', {})
                        if metric == 'std_deviation':
                            field_result['std_deviation'] = ext_stats.get('std_deviation')
                        else:
                            field_result['variance'] = ext_stats.get('variance')
                    elif metric in ['median', 'q1', 'q3', 'q5']:
                        percentiles = field_aggs.get('percentiles', {}).get('values', {})
                        key = '50.0' if metric == 'median' else '25.0' if metric == 'q1' else '75.0' if metric == 'q3' else '5.0'
                        field_result[metric] = percentiles.get(key)
                    elif metric == 'mode':
                        buckets = field_aggs.get('mode', {}).get('buckets', [])
                        if buckets:
                            field_result['mode'] = buckets[0].get('key')
                            field_result['mode_count'] = buckets[0].get('doc_count')

                result[field] = field_result

            return result

        except Exception as e:
            return {'error': f'结果处理错误: {str(e)}'}

    def process_distribution_result(self, es_result: Dict, original_config: Dict) -> Dict:
        """处理分布分析结果，支持百分比计算"""
        try:
            aggregations = es_result.get('aggregations', {})
            config = original_config['query']['config']

            # 获取总计数用于百分比计算
            total_count = aggregations.get('_total_count', {}).get('value', 0)

            return self._process_distribution_aggregations(
                aggregations, config, total_count, level=0
            )

        except Exception as e:
            return {'error': f'分布结果处理错误: {str(e)}'}

    def _process_distribution_aggregations(self, aggs: Dict, config: Dict,
                                           parent_total: int, level: int = 0) -> Dict:
        """递归处理分布分析聚合结果"""
        result = {'buckets': []}

        # 获取当前层级的聚合键
        aggregation_keys = [k for k in aggs.keys() if not k.startswith('_')]

        for agg_key in aggregation_keys:
            agg_data = aggs[agg_key]

            if 'buckets' in agg_data:
                # 处理桶聚合
                buckets = agg_data['buckets']
                current_level_total = sum(bucket.get('doc_count', 0) for bucket in buckets)

                for bucket in buckets:
                    bucket_result = self._process_bucket(bucket, config, parent_total, current_level_total, level)

                    # 递归处理子聚合
                    sub_aggs = {k: v for k, v in bucket.items()
                                if k not in ['key', 'from', 'to', 'doc_count', 'key_as_string']}

                    if sub_aggs and level < 5:  # 防止无限递归
                        sub_result = self._process_distribution_aggregations(
                            sub_aggs, config, bucket.get('doc_count', 0), level + 1
                        )
                        if sub_result.get('buckets'):
                            bucket_result['sub_aggregations'] = sub_result

                    result['buckets'].append(bucket_result)

        return result

    def _process_bucket(self, bucket: Dict, config: Dict,
                        parent_total: int, current_level_total: int, level: int) -> Dict:
        """处理单个桶的结果"""
        bucket_result = {
            'key': bucket.get('key'),
            'key_as_string': bucket.get('key_as_string'),
            'from': bucket.get('from'),
            'to': bucket.get('to'),
            'doc_count': bucket.get('doc_count', 0)
        }

        metrics = config.get('metrics', ['count', 'percentage'])
        metrics_field = config.get('metrics_field')

        # 计算指标
        metrics_result = {}

        for metric in metrics:
            if metric == 'count':
                metrics_result['count'] = bucket.get('doc_count', 0)

            elif metric == 'percentage':
                # 计算百分比：当前桶计数 / 父级总计数 * 100
                if parent_total > 0:
                    percentage = (bucket.get('doc_count', 0) / parent_total) * 100
                    metrics_result['percentage'] = round(percentage, 2)
                else:
                    metrics_result['percentage'] = 0.0

            elif metric in ['avg', 'sum', 'min', 'max'] and metrics_field:
                metric_value = bucket.get(metric, {}).get('value')
                if metric_value is not None:
                    metrics_result[metric] = metric_value

        if metrics_result:
            bucket_result['metrics'] = metrics_result

        return bucket_result


def demo_enhanced_distribution():
    """演示分布分析功能"""
    translator = OpenSearchStatsTranslator()

    # 测试用例：不同部门、年龄段的薪资分布
    test_query = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["education"],
                "groups": ["department"],
                "buckets": [
                    {
                        "type": "range",
                        "field": "age",
                        "ranges": [
                            {"key": "20-30", "from": 20, "to": 30},
                            {"key": "30-40", "from": 30, "to": 40}
                        ]
                    }
                ],
                "metrics": ["count", "percentage", "avg"],
                "metrics_field": "salary",
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

    # 生成DSL
    dsl = translator.translate(test_query)
    print("1. 生成的DSL（包含百分比计算）:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 模拟OpenSearch返回结果
    mock_result = {
        "aggregations": {
            "_total_count": {"value": 1000},
            "department": {
                "buckets": [
                    {
                        "key": "engineering",
                        "doc_count": 600,
                        "_group_count": {"value": 600},
                        "age": {
                            "buckets": [
                                {
                                    "key": "20-30",
                                    "from": 20,
                                    "to": 30,
                                    "doc_count": 300,
                                    "_bucket_count": {"value": 300},
                                    "education": {
                                        "buckets": [
                                            {
                                                "key": "bachelor",
                                                "doc_count": 200,
                                                "_dimension_count": {"value": 200},
                                                "count": {"value": 200},
                                                "avg": {"value": 15000}
                                            },
                                            {
                                                "key": "master",
                                                "doc_count": 100,
                                                "_dimension_count": {"value": 100},
                                                "count": {"value": 100},
                                                "avg": {"value": 20000}
                                            }
                                        ]
                                    }
                                },
                                {
                                    "key": "30-40",
                                    "from": 30,
                                    "to": 40,
                                    "doc_count": 300,
                                    "_bucket_count": {"value": 300},
                                    "education": {
                                        "buckets": [
                                            {
                                                "key": "bachelor",
                                                "doc_count": 180,
                                                "_dimension_count": {"value": 180},
                                                "count": {"value": 180},
                                                "avg": {"value": 25000}
                                            },
                                            {
                                                "key": "master",
                                                "doc_count": 120,
                                                "_dimension_count": {"value": 120},
                                                "count": {"value": 120},
                                                "avg": {"value": 30000}
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "key": "sales",
                        "doc_count": 400,
                        "_group_count": {"value": 400},
                        "age": {
                            "buckets": [
                                {
                                    "key": "20-30",
                                    "from": 20,
                                    "to": 30,
                                    "doc_count": 200,
                                    "_bucket_count": {"value": 200},
                                    "education": {
                                        "buckets": [
                                            {
                                                "key": "bachelor",
                                                "doc_count": 150,
                                                "_dimension_count": {"value": 150},
                                                "count": {"value": 150},
                                                "avg": {"value": 12000}
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }

    print("\n2. 模拟的OpenSearch返回结果:")
    print(json.dumps(mock_result, indent=2, ensure_ascii=False))

    # 处理结果
    processed_result = translator.process_distribution_result(mock_result, test_query)
    print("\n3. 处理后的分布分析结果（包含百分比）:")
    print(json.dumps(processed_result, indent=2, ensure_ascii=False))


def test_basic_stats_query():
    """测试基础统计查询功能"""
    print("=== 测试1: 基础统计查询 ===")

    translator = OpenSearchStatsTranslator()

    # 基础统计查询
    basic_stats = {
        "query": {
            "type": "stats",
            "config": {
                "fields": ["price", "quantity"],
                "metrics": ["min", "max", "avg", "count"]
            }
        }
    }

    dsl = translator.translate(basic_stats)
    print("生成的DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 模拟返回结果
    mock_result = {
        "aggregations": {
            "price": {
                "min": {"value": 10},
                "max": {"value": 100},
                "avg": {"value": 55.5},
                "count": {"value": 50}
            },
            "quantity": {
                "min": {"value": 1},
                "max": {"value": 20},
                "avg": {"value": 8.5},
                "count": {"value": 50}
            }
        }
    }

    result = translator.process_stats_result(mock_result, basic_stats)
    print("\n处理后的统计结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 验证结果
    assert "price" in result
    assert result["price"]["min"] == 10
    assert result["price"]["max"] == 100
    assert result["price"]["avg"] == 55.5
    print("✓ 基础统计查询测试通过")


def test_stats_with_filters():
    """测试带过滤条件的统计查询"""
    print("\n=== 测试2: 带过滤的统计查询 ===")

    translator = OpenSearchStatsTranslator()

    stats_with_filters = {
        "query": {
            "type": "stats",
            "config": {
                "fields": ["salary"],
                "metrics": ["min", "max", "avg", "median", "std_deviation"],
                "filters": [
                    {
                        "field": "department",
                        "operator": "eq",
                        "value": "engineering"
                    },
                    {
                        "field": "age",
                        "operator": "gte",
                        "value": 25
                    },
                    {
                        "field": "salary",
                        "operator": "lt",
                        "value": 100000
                    }
                ]
            }
        }
    }

    dsl = translator.translate(stats_with_filters)
    print("生成的DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 验证过滤条件是否正确转换
    query = dsl.get("query", {})
    assert "bool" in query
    assert "must" in query["bool"]
    assert len(query["bool"]["must"]) == 3
    print("✓ 过滤条件转换正确")


def test_basic_distribution():
    """测试基础分布分析"""
    print("\n=== 测试3: 基础分布分析 ===")

    translator = OpenSearchStatsTranslator()

    basic_dist = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["category"],
                "metrics": ["count", "percentage"]
            }
        }
    }

    dsl = translator.translate(basic_dist)
    print("生成的DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 模拟简单分布结果
    mock_result = {
        "aggregations": {
            "_total_count": {"value": 1000},
            "category": {
                "buckets": [
                    {
                        "key": "electronics",
                        "doc_count": 400,
                        "_dimension_count": {"value": 400},
                        "count": {"value": 400}
                    },
                    {
                        "key": "books",
                        "doc_count": 350,
                        "_dimension_count": {"value": 350},
                        "count": {"value": 350}
                    },
                    {
                        "key": "clothing",
                        "doc_count": 250,
                        "_dimension_count": {"value": 250},
                        "count": {"value": 250}
                    }
                ]
            }
        }
    }

    result = translator.process_distribution_result(mock_result, basic_dist)
    print("\n处理后的分布结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 验证百分比计算
    buckets = result.get("buckets", [])
    for bucket in buckets:
        metrics = bucket.get("metrics", {})
        if metrics.get("count") == 400:
            assert metrics.get("percentage") == 40.0  # 400/1000 * 100
        elif metrics.get("count") == 350:
            assert metrics.get("percentage") == 35.0
    print("✓ 基础分布分析测试通过")


def test_terms_bucket_distribution():
    """测试术语桶分布分析"""
    print("\n=== 测试4: 术语桶分布分析 ===")

    translator = OpenSearchStatsTranslator()

    terms_dist = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["sub_category"],
                "buckets": [
                    {
                        "type": "terms",
                        "field": "main_category",
                        "size": 5
                    }
                ],
                "metrics": ["count", "percentage", "avg"],
                "metrics_field": "price"
            }
        }
    }

    dsl = translator.translate(terms_dist)
    print("生成的DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 验证术语桶配置
    aggs = dsl.get("aggs", {})
    assert "main_category" in aggs
    assert aggs["main_category"]["terms"]["size"] == 5
    print("✓ 术语桶配置正确")


def test_range_bucket_distribution():
    """测试范围桶分布分析"""
    print("\n=== 测试5: 范围桶分布分析 ===")

    translator = OpenSearchStatsTranslator()

    range_dist = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["status"],
                "buckets": [
                    {
                        "type": "range",
                        "field": "price",
                        "ranges": [
                            {"key": "低价", "from": 0, "to": 100},
                            {"key": "中价", "from": 100, "to": 500},
                            {"key": "高价", "from": 500}
                        ]
                    }
                ],
                "metrics": ["count", "percentage"]
            }
        }
    }

    dsl = translator.translate(range_dist)
    print("生成的DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 验证范围桶配置
    aggs = dsl.get("aggs", {})
    assert "price" in aggs
    assert len(aggs["price"]["range"]["ranges"]) == 3
    print("✓ 范围桶配置正确")


def test_complex_metrics():
    """测试复杂指标计算"""
    print("\n=== 测试6: 复杂指标计算 ===")

    translator = OpenSearchStatsTranslator()

    complex_stats = {
        "query": {
            "type": "stats",
            "config": {
                "fields": ["score"],
                "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation", "variance", "mode"]
            }
        }
    }

    dsl = translator.translate(complex_stats)
    print("生成的DSL:")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    # 模拟包含复杂指标的结果
    mock_result = {
        "aggregations": {
            "score": {
                "min": {"value": 0},
                "max": {"value": 100},
                "avg": {"value": 75.5},
                "percentiles": {
                    "values": {
                        "25.0": 60.0,
                        "50.0": 75.0,
                        "75.0": 90.0
                    }
                },
                "extended_stats": {
                    "std_deviation": 15.2,
                    "variance": 231.04
                },
                "mode": {
                    "buckets": [
                        {"key": 80, "doc_count": 25}
                    ]
                }
            }
        }
    }

    result = translator.process_stats_result(mock_result, complex_stats)
    print("\n处理后的复杂指标结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 验证复杂指标
    score_result = result.get("score", {})
    assert score_result["min"] == 0
    assert score_result["max"] == 100
    assert score_result["avg"] == 75.5
    assert score_result["q1"] == 60.0
    assert score_result["median"] == 75.0
    assert score_result["q3"] == 90.0
    assert score_result["std_deviation"] == 15.2
    assert score_result["variance"] == 231.04
    assert score_result["mode"] == 80
    print("✓ 复杂指标计算测试通过")


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试7: 错误处理 ===")

    translator = OpenSearchStatsTranslator()

    # 测试无效查询类型
    invalid_query = {
        "query": {
            "type": "invalid_type",
            "config": {
                "fields": ["test"]
            }
        }
    }

    result = translator.translate(invalid_query)
    assert "error" in result
    print("✓ 无效查询类型错误处理正确")

    # 测试缺少必要字段
    missing_fields = {
        "query": {
            "type": "stats",
            "config": {
                # 缺少fields字段
            }
        }
    }

    result = translator.translate(missing_fields)
    # 应该能正常处理空字段列表
    assert "aggs" in result
    print("✓ 缺失字段处理正确")


def test_percentage_calculation_edge_cases():
    """测试百分比计算的边界情况"""
    print("\n=== 测试8: 百分比计算边界情况 ===")

    translator = OpenSearchStatsTranslator()

    # 测试除零情况
    mock_zero_result = {
        "aggregations": {
            "_total_count": {"value": 0},  # 总数为0
            "category": {
                "buckets": [
                    {
                        "key": "test",
                        "doc_count": 0,
                        "_dimension_count": {"value": 0},
                        "count": {"value": 0}
                    }
                ]
            }
        }
    }

    test_query = {
        "query": {
            "type": "distribution",
            "config": {
                "dimensions": ["category"],
                "metrics": ["count", "percentage"]
            }
        }
    }

    result = translator.process_distribution_result(mock_zero_result, test_query)
    buckets = result.get("buckets", [])
    if buckets:
        metrics = buckets[0].get("metrics", {})
        assert metrics.get("percentage") == 0.0  # 除零时应返回0

    print("✓ 除零情况处理正确")


def run_all_tests():
    """运行所有测试"""
    print("开始运行OpenSearch翻译器测试...\n")

    try:
        test_basic_stats_query()
        test_stats_with_filters()
        test_basic_distribution()
        test_terms_bucket_distribution()
        test_range_bucket_distribution()
        test_complex_metrics()
        test_error_handling()
        test_percentage_calculation_edge_cases()

        print("\n🎉 所有测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行演示
    demo_enhanced_distribution()

    print("\n" + "=" * 60)
    # 运行测试
    run_all_tests()


