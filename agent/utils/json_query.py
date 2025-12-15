import json
from typing import Dict, Any
import time

from agent.utils.json_2_dsl import OpenSearchQueryTranslator
from agent.utils.search_tools.opensearch_connection import search_by_dsl


def test_descriptive_stats():
    """测试描述性统计查询"""
    print("\n" + "=" * 60)
    print("测试1: 描述性统计")
    print("=" * 60)

    query = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "fields": ["patient_age", "diabetes_time_y"],
                "filters": [
                    {
                        "field": "diabetic_retinopathy",
                        "operator": "eq",
                        "value": 1
                    },
                    {
                        "field": "patient_age",
                        "operator": "gte",
                        "value": 40
                    }
                ],
                "metrics": ["count", "min", "max", "avg", "sum", "q1", "median", "q3"]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl = translator.translate(query)
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result = search_by_dsl(dsl, index="brset", return_whole_response=True)

    print("\n查询结果:")
    print(f"查询耗时: {result.get('took', 0)}ms")
    print(f"匹配文档数: {result.get('hits', {}).get('total', {}).get('value', 0)}")

    formatted = translator.format_result("descriptive_stats", result)
    print("\n格式化结果:")
    print(json.dumps(formatted, indent=2, ensure_ascii=False))

    return result


def test_complete_stats():
    """测试完整统计查询"""
    print("\n" + "=" * 60)
    print("测试2: 完整统计")
    print("=" * 60)

    query = {
        "query": {
            "type": "complete_stats",
            "config": {
                "fields": ["patient_age", "diabetes_time_y"],
                "metrics": ["count", "min", "max", "avg", "sum", "q1", "q5", "median", "q3", "std_deviation",
                            "variance", "mode", "cardinality"],
                "filters": [
                    {
                        "field": "drusens",
                        "operator": "eq",
                        "value": 0
                    },
                    {
                        "field": "patient_age",
                        "operator": "gte",
                        "value": 30
                    }
                ]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl = translator.translate(query)
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result = search_by_dsl(dsl, index="brset", return_whole_response=True)

    print("\n查询结果:")
    print(f"查询耗时: {result.get('took', 0)}ms")

    formatted = translator.format_result("complete_stats", result)
    print("\n格式化结果:")
    print(json.dumps(formatted, indent=2, ensure_ascii=False))

    return result


def test_frequency_analysis():
    """测试频率分析"""
    print("\n" + "=" * 60)
    print("测试3: 频率分析")
    print("=" * 60)

    query = {
        "query": {
            "type": "frequency_analysis",
            "config": {
                "fields": ["drusens", "optic_disc", "camera.keyword"],
                "group_by": ["diabetic_retinopathy", "patient_sex"],
                "metrics": ["count", "percentage"],
                "filters": [
                    {
                        "field": "quality",
                        "operator": "eq",
                        "value": "Adequate"
                    },
                    {
                        "field": "patient_age",
                        "operator": "gte",
                        "value": 20
                    }
                ]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl = translator.translate(query)
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result = search_by_dsl(dsl, index="brset", return_whole_response=True)

    print("\n查询结果:")
    print(f"查询耗时: {result.get('took', 0)}ms")

    formatted = translator.format_result("frequency_analysis", result)
    print("\n格式化结果:")
    print(json.dumps(formatted, indent=2, ensure_ascii=False))

    return result


def test_cross_analysis():
    """测试交叉分析"""
    print("\n" + "=" * 60)
    print("测试4: 交叉分析")
    print("=" * 60)

    query = {
        "query": {
            "type": "cross_analysis",
            "config": {
                "fields": ["hemorrhage", "macular_edema"],
                "group_by": ["patient_sex"],
                "bucket_ranges": [
                    {
                        "field": "patient_age",
                        "ranges": [
                            {"key": "young", "from": 0, "to": 20},
                            {"key": "adult", "from": 20, "to": 40},
                            {"key": "middle", "from": 40, "to": 60},
                            {"key": "old", "from": 60}
                        ],
                        "type": "range"
                    }
                ],
                "metrics": ["count", "percentage"],
                "filters": [
                    {
                        "field": "diabetes",
                        "operator": "eq",
                        "value": "yes"
                    },
                    {
                        "field": "diabetes_time_y",
                        "operator": "gte",
                        "value": 5
                    }
                ]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl = translator.translate(query)
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result = search_by_dsl(dsl, index="brset", return_whole_response=True)

    print("\n查询结果:")
    print(f"查询耗时: {result.get('took', 0)}ms")

    formatted = translator.format_result("cross_analysis", result)
    print("\n格式化结果:")
    print(json.dumps(formatted, indent=2, ensure_ascii=False))

    return result


def test_range_analysis():
    """测试范围分析"""
    print("\n" + "=" * 60)
    print("测试5: 范围分析")
    print("=" * 60)

    query = {
        "query": {
            "type": "range_analysis",
            "config": {
                "field": "patient_age",
                "ranges": [
                    {"key": "0-20", "from": 0, "to": 20},
                    {"key": "20-40", "from": 20, "to": 40},
                    {"key": "40-60", "from": 40, "to": 60},
                    {"key": "60+", "from": 60}
                ],
                "group_by": ["diabetic_retinopathy"],
                "metrics_field": "diabetes_time_y",
                "metrics": ["count", "avg"],
                "filters": [
                    {
                        "field": "diabetes",
                        "operator": "eq",
                        "value": "yes"
                    }
                ]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl = translator.translate(query)
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result = search_by_dsl(dsl, index="brset", return_whole_response=True)

    print("\n查询结果:")
    print(f"查询耗时: {result.get('took', 0)}ms")

    formatted = translator.format_result("range_analysis", result)
    print("\n格式化结果:")
    print(json.dumps(formatted, indent=2, ensure_ascii=False))

    return result


def test_exists_and_regex():
    """测试exists和正则表达式查询"""
    print("\n" + "=" * 60)
    print("测试6: exists和正则查询")
    print("=" * 60)

    # 测试exists
    query1 = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "fields": ["patient_age"],
                "filters": [
                    {
                        "field": "image_id",
                        "operator": "exists"
                    },
                    {
                        "field": "patient_id",
                        "operator": "exists"
                    }
                ]
            }
        }
    }

    print("测试6.1: exists查询")
    print(json.dumps(query1, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl1 = translator.translate(query1)
    print(json.dumps(dsl1, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result1 = search_by_dsl(dsl1, index="brset", return_whole_response=True)
    print(f"查询耗时: {result1.get('took', 0)}ms")

    # 测试正则表达式
    query2 = {
        "query": {
            "type": "frequency_analysis",
            "config": {
                "fields": ["camera.keyword"],
                "filters": [
                    {
                        "field": "patient_id",
                        "operator": "regexp",
                        "value": ".*[0-9]$"
                    }
                ]
            }
        }
    }

    print("\n\n测试6.2: 正则表达式查询")
    print(json.dumps(query2, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    dsl2 = translator.translate(query2)
    print(json.dumps(dsl2, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result2 = search_by_dsl(dsl2, index="brset", return_whole_response=True)
    print(f"查询耗时: {result2.get('took', 0)}ms")

    return result1, result2


def test_pagination():
    """测试分页功能"""
    print("\n" + "=" * 60)
    print("测试7: 分页查询")
    print("=" * 60)

    query = {
        "query": {
            "type": "frequency_analysis",
            "config": {
                "fields": ["camera.keyword"],
                "filters": [
                    {
                        "field": "patient_age",
                        "operator": "gte",
                        "value": 30
                    }
                ]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))

    translator = OpenSearchQueryTranslator(index_name="brset")

    for page in [1, 2]:
        print(f"\n第{page}页 (每页5条):")
        dsl = translator.translate_with_pagination(query, page=page, size=5)
        print("生成的OpenSearch DSL (带分页):")
        print(json.dumps(dsl, indent=2, ensure_ascii=False))

        print(f"\n执行第{page}页查询...")
        result = search_by_dsl(dsl, index="brset", return_whole_response=True)

        hits = result.get("hits", {}).get("hits", [])
        print(f"第{page}页结果数量: {len(hits)}")

        if hits:
            print("前5条结果:")
            for i, hit in enumerate(hits[:5], 1):
                source = hit.get("_source", {})
                print(
                    f"{i}. image_id: {source.get('image_id')}, age: {source.get('patient_age')}, camera: {source.get('camera')}")

    return result


def test_complex_filters():
    """测试复杂过滤条件"""
    print("\n" + "=" * 60)
    print("测试8: 复杂过滤条件")
    print("=" * 60)

    query = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "fields": ["patient_age", "diabetes_time_y"],
                "filters": [
                    {
                        "field": "nationality.keyword",
                        "operator": "eq",
                        "value": "Brazil"
                    },
                    {
                        "field": "patient_age",
                        "operator": "range",
                        "value": {
                            "gte": 20,
                            "lte": 60
                        }
                    },
                    {
                        "field": "diabetes_time_y",
                        "operator": "gt",
                        "value": 5
                    },
                    {
                        "field": "patient_sex",
                        "operator": "in",
                        "value": [1, 2]
                    },
                    {
                        "field": "camera.keyword",
                        "operator": "wildcard",
                        "value": "Canon*"
                    }
                ],
                "metrics": ["count", "min", "max", "avg", "sum"]
            }
        }
    }

    print("查询JSON:")
    print(json.dumps(query, indent=2, ensure_ascii=False))
    print("\n生成的OpenSearch DSL:")

    translator = OpenSearchQueryTranslator(index_name="brset")
    dsl = translator.translate(query)
    print(json.dumps(dsl, indent=2, ensure_ascii=False))

    print("\n执行查询...")
    result = search_by_dsl(dsl, index="brset", return_whole_response=True)

    print("\n查询结果:")
    print(f"查询耗时: {result.get('took', 0)}ms")

    formatted = translator.format_result("descriptive_stats", result)
    print("\n格式化结果:")
    print(json.dumps(formatted, indent=2, ensure_ascii=False))

    return result


def test_error_cases():
    """测试错误情况"""
    print("\n" + "=" * 60)
    print("测试9: 错误情况处理")
    print("=" * 60)

    translator = OpenSearchQueryTranslator(index_name="brset")

    # 测试1: 缺少必要字段
    print("测试9.1: 缺少必要字段")
    bad_query1 = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "filters": []
            }
        }
    }

    try:
        dsl1 = translator.translate(bad_query1)
        print("应该抛出异常，但没有抛出")
    except ValueError as e:
        print(f"预期异常: {e}")

    # 测试2: 无效的查询类型
    print("\n测试9.2: 无效的查询类型")
    bad_query2 = {
        "query": {
            "type": "invalid_type",
            "config": {
                "fields": ["patient_age"]
            }
        }
    }

    try:
        dsl2 = translator.translate(bad_query2)
        print("应该抛出异常，但没有抛出")
    except ValueError as e:
        print(f"预期异常: {e}")

    # 测试3: 无效的操作符
    print("\n测试9.3: 无效的操作符")
    bad_query3 = {
        "query": {
            "type": "descriptive_stats",
            "config": {
                "fields": ["patient_age"],
                "filters": [
                    {
                        "field": "patient_age",
                        "operator": "invalid_operator",
                        "value": 30
                    }
                ]
            }
        }
    }

    try:
        dsl3 = translator.translate(bad_query3)
        result3 = search_by_dsl(dsl3, index="brset", return_whole_response=False)
        print(f"查询结果: {result3}")
    except Exception as e:
        print(f"查询异常: {e}")


def run_all_tests():
    """运行所有测试"""
    print("开始测试OpenSearch查询翻译器...")
    print("=" * 60)

    all_results = {}

    try:
        # 测试1: 描述性统计
        all_results["descriptive_stats"] = test_descriptive_stats()

        # 测试2: 完整统计
        all_results["complete_stats"] = test_complete_stats()

        # 测试3: 频率分析
        all_results["frequency_analysis"] = test_frequency_analysis()

        # 测试4: 交叉分析
        all_results["cross_analysis"] = test_cross_analysis()

        # 测试5: 范围分析
        all_results["range_analysis"] = test_range_analysis()

        # 测试6: exists和正则
        all_results["exists_regex"] = test_exists_and_regex()

        # 测试7: 分页
        all_results["pagination"] = test_pagination()

        # 测试8: 复杂过滤
        all_results["complex_filters"] = test_complex_filters()

        # 测试9: 错误情况
        test_error_cases()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

        # 生成测试报告
        print("\n测试报告:")
        for test_name, result in all_results.items():
            if isinstance(result, tuple):
                print(f"{test_name}: 完成")
            elif result:
                took = result.get('took', 0) if isinstance(result, dict) else 0
                print(f"{test_name}: 完成 (耗时: {took}ms)")
            else:
                print(f"{test_name}: 失败")

    except Exception as e:
        print(f"测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()

    return all_results


if __name__ == "__main__":
    # 运行所有测试
    run_all_tests()