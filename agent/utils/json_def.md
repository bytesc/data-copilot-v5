# OpenSearch Statistical Analysis JSON Interface Specification

## Overview

This document defines the JSON interface format for OpenSearch statistical analysis. This interface allows users to perform complex data statistical analysis through simple JSON configuration, including descriptive statistics, frequency analysis, cross-analysis, and range analysis.

---

## 1. Basic Structure

### JSON Root Structure
```json
{
  "query": {
    "type": "string",        // Required: Query type
    "config": {              // Required: Query configuration
      "fields": [],
      "group_by": [],
      "filters": [],
      "metrics": [],
      "bucket_ranges": [],
      "ranges": [],
      "field": ""
    }
  }
}
```

### Field Definitions
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query.type` | string | Yes | Query type, supported values in Section 2 |
| `query.config` | object | Yes | Query configuration object |
| `query.config.fields` | string[] | Conditional | List of fields to analyze (depends on query type) |
| `query.config.group_by` | string[] | No | List of grouping fields |
| `query.config.filters` | object[] | No | Array of filter conditions |
| `query.config.metrics` | string[] | No | Array of statistical metrics |
| `query.config.bucket_ranges` | object[] | No | Custom bucket ranges for range analysis |
| `query.config.ranges` | object[] | Conditional | Range definitions for range_analysis type |
| `query.config.field` | string | Conditional | Target field for range_analysis type |
| `query.config.metrics_field` | string | No | Field for metric calculations |

---

## 2. Query Types (query.type)

### 2.1 Available Types

| Type | Value | Description |
|------|-------|-------------|
| Descriptive Statistics | `"descriptive_stats"` | Basic statistical analysis for numerical fields (min, max, median, Q1, Q3, etc.) |
| Complete Statistics | `"complete_stats"` | Comprehensive statistical analysis including all metrics (Q5, std_deviation, variance, mode, etc.) |
| Frequency Analysis | `"frequency_analysis"` | Frequency analysis for categorical fields |
| Cross Analysis | `"cross_analysis"` | Combined analysis supporting grouped statistics, custom buckets, and percentages |
| Range Analysis | `"range_analysis"` | Analysis with custom range buckets (e.g., age groups) |

### 2.2 Type Selection Guide
- **Basic numerical field analysis** → Use `descriptive_stats`
- **Comprehensive numerical analysis with all percentiles** → Use `complete_stats`
- **Categorical field distribution** → Use `frequency_analysis`
- **Grouped comparative analysis with custom buckets** → Use `cross_analysis`
- **Custom range grouping (e.g., age groups)** → Use `range_analysis`

---

## 3. Configuration Parameters (query.config)

### 3.1 fields Parameter

#### Definition
```json
"fields": ["field1", "field2", ...]
```

#### Description
- **Type**: String array
- **Required**: Yes (for descriptive_stats, complete_stats, frequency_analysis, cross_analysis)
- **Required**: No (for range_analysis)
- **Purpose**: Specifies field names to analyze
- **Constraints**: Must contain at least one field (where required)

#### Example
```json
"fields": ["age", "income", "height"]
```

### 3.2 group_by Parameter

#### Definition
```json
"group_by": ["group_field1", "group_field2", ...]
```

#### Description
- **Type**: String array
- **Required**: No
- **Purpose**: Specifies grouping fields for grouped statistics
- **Hierarchy**: Supports multi-level grouping, nested in array order
- **Maximum Depth**: Recommended not to exceed 3 levels

#### Example
```json
"group_by": ["region", "city", "district"]  // Three-level grouping
```

### 3.3 filters Parameter

#### Definition
```json
"filters": [
  {
    "field": "string",      // Field name
    "operator": "string",   // Operator
    "value": "any"         // Value (type depends on operator)
  }
]
```

#### 3.3.1 Supported Operators (operator)

| Operator | Value | Description | Value Type | Example |
|----------|-------|-------------|------------|---------|
| Equals | `"eq"` | Field equals specified value | Any | `"value": "male"` |
| Not equals | `"neq"` | Field does not equal specified value | Any | `"value": "female"` |
| Greater than | `"gt"` | Field greater than specified value | Number/Date | `"value": 18` |
| Greater than or equal | `"gte"` | Field greater than or equal to specified value | Number/Date | `"value": "2023-01-01"` |
| Less than | `"lt"` | Field less than specified value | Number/Date | `"value": 100` |
| Less than or equal | `"lte"` | Field less than or equal to specified value | Number/Date | `"value": "2023-12-31"` |
| In | `"in"` | Field value is in specified list | Array | `"value": ["A", "B", "C"]` |
| Range | `"range"` | Field is within specified range | Object | `"value": {"gte": 10, "lte": 20}` |
| Wildcard match | `"like"` | SQL LIKE pattern matching | String | `"value": "张%"` |
| Exists | `"exists"` | Field exists (not null) | None | No value needed |
| Missing | `"missing"` | Field is missing or null | None | No value needed |
| Wildcard | `"wildcard"` | Wildcard pattern matching | String | `"value": "张*"` |
| Regexp | `"regexp"` | Regular expression matching | String | `"value": "张.*"` |

#### 3.3.2 Filter Examples

**Single condition filter:**
```json
{
  "field": "age",
  "operator": "gte",
  "value": 18
}
```

**Multiple conditions:**
```json
"filters": [
  {
    "field": "gender",
    "operator": "eq",
    "value": "male"
  },
  {
    "field": "income",
    "operator": "gte",
    "value": 5000
  },
  {
    "field": "city",
    "operator": "in",
    "value": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"]
  },
  {
    "field": "name",
    "operator": "exists"
  }
]
```

**Date range filter:**
```json
{
  "field": "create_time",
  "operator": "range",
  "value": {
    "gte": "2023-01-01T00:00:00",
    "lte": "2023-12-31T23:59:59"
  }
}
```

### 3.4 metrics Parameter

#### Definition
```json
"metrics": ["metric1", "metric2", ...]
```

#### Supported Metrics List

| Metric | Value | Description | Applicable Types | Available in Types |
|--------|-------|-------------|------------------|-------------------|
| Count | `"count"` | Number of documents | All types | All |
| Cardinality | `"cardinality"` | Number of unique values | Categorical fields | descriptive_stats, complete_stats, cross_analysis |
| Minimum | `"min"` | Minimum value | Numerical fields | descriptive_stats, complete_stats |
| Maximum | `"max"` | Maximum value | Numerical fields | descriptive_stats, complete_stats |
| Average | `"avg"` | Average value | Numerical fields | descriptive_stats, complete_stats, range_analysis |
| Sum | `"sum"` | Sum of values | Numerical fields | descriptive_stats, complete_stats, range_analysis |
| Percentage | `"percentage"` | Percentage of total | Grouped statistics | cross_analysis, frequency_analysis |
| Median | `"median"` | Median (50th percentile) | Numerical fields | descriptive_stats, complete_stats |
| First quartile | `"q1"` | First quartile (25th percentile) | Numerical fields | descriptive_stats, complete_stats |
| Fifth percentile | `"q5"` | Fifth percentile | Numerical fields | complete_stats |
| Third quartile | `"q3"` | Third quartile (75th percentile) | Numerical fields | descriptive_stats, complete_stats |
| Standard deviation | `"std_deviation"` | Standard deviation | Numerical fields | complete_stats |
| Variance | `"variance"` | Variance | Numerical fields | complete_stats |
| Mode | `"mode"` | Most frequent value | All types | complete_stats |

#### Default Metrics by Query Type
- **descriptive_stats**: `["min", "max", "avg", "count", "q1", "median", "q3"]`
- **complete_stats**: All available metrics
- **frequency_analysis**: `["count", "percentage"]`
- **cross_analysis**: `["count", "percentage"]`
- **range_analysis**: `["count"]`

### 3.5 bucket_ranges Parameter (for cross_analysis)

#### Definition
```json
"bucket_ranges": [
  {
    "field": "string",      // Field name for range bucket
    "ranges": [            // Range definitions
      {
        "key": "string",   // Range label
        "from": number,    // Start value (inclusive)
        "to": number       // End value (exclusive)
      }
    ],
    "type": "range"        // Always "range"
  }
]
```

#### Example
```json
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
]
```

### 3.6 ranges and field Parameters (for range_analysis)

#### Definition
```json
{
  "field": "age",  // Field to create ranges on
  "ranges": [     // Range definitions
    {"key": "0-20", "from": 0, "to": 20},
    {"key": "20-40", "from": 20, "to": 40}
  ],
  "metrics_field": "income"  // Optional: Field for metric calculations
}
```

#### Description
- **field**: Required for range_analysis, the field to create custom ranges on
- **ranges**: Required for range_analysis, array of range definitions
- **metrics_field**: Optional, field to calculate metrics (avg, sum) within each range

---

## 4. Query Examples

### 4.1 Descriptive Statistics

**Function**: Basic statistical analysis of numerical fields

```json
{
  "query": {
    "type": "descriptive_stats",
    "config": {
      "fields": ["age", "salary", "work_years"],
      "filters": [
        {
          "field": "department",
          "operator": "eq",
          "value": "engineering"
        },
        {
          "field": "hire_date",
          "operator": "gte",
          "value": "2020-01-01"
        }
      ]
    }
  }
}
```

**Output Includes**: count, min, max, avg, sum, q1, median, q3, std_deviation, variance, mode, cardinality

### 4.2 Complete Statistics

**Function**: Comprehensive statistical analysis with all percentiles

```json
{
  "query": {
    "type": "complete_stats",
    "config": {
      "fields": ["age", "blood_pressure", "cholesterol"],
      "metrics": ["min", "max", "avg", "median", "q1", "q5", "q3", "std_deviation", "mode"],
      "filters": [
        {
          "field": "gender",
          "operator": "eq",
          "value": "male"
        },
        {
          "field": "age",
          "operator": "range",
          "value": {"gte": 18, "lte": 80}
        }
      ]
    }
  }
}
```

**Output Includes**: All specified metrics including Q5 (5th percentile)

### 4.3 Frequency Analysis

**Function**: Analyze distribution of categorical fields

```json
{
  "query": {
    "type": "frequency_analysis",
    "config": {
      "fields": ["education_level", "job_title"],
      "group_by": ["department"],
      "filters": [
        {
          "field": "active",
          "operator": "eq",
          "value": true
        }
      ],
      "metrics": ["count", "percentage"]
    }
  }
}
```

**Output Description**:
- Grouped by department
- Within each department, statistics for education levels and job titles
- Includes count and percentage

### 4.4 Cross Analysis with Custom Buckets

**Function**: Advanced grouped analysis with custom range buckets

```json
{
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
          "operator": "gte",
          "value": "2023-01-01"
        }
      ]
    }
  }
}
```

**Analysis Flow**:
1. First group by age ranges (young, middle, old)
2. Within each age range, group by gender
3. Within each gender group, analyze disease status and treatment type
4. Calculate count and percentage for each combination

### 4.5 Range Analysis (Age Group Disease Ratio)

**Function**: Analyze disease ratios by custom age groups

```json
{
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
```

**Output Structure**:
- Each age group (0-20, 20-40, etc.)
- Within each group, breakdown by disease status and disease type
- For each subgroup: count and average age

---

## 5. Response Format

### 5.1 Descriptive/Complete Statistics Response
```json
{
  "age": {
    "count": 1000,
    "min": 18,
    "max": 80,
    "avg": 45.5,
    "sum": 45500,
    "q1": 30.0,
    "q5": 25.0,
    "median": 45.0,
    "q3": 60.0,
    "std_deviation": 15.2,
    "variance": 231.04,
    "mode": 35,
    "mode_count": 120,
    "cardinality": 63
  }
}
```

### 5.2 Frequency/Cross Analysis Response
```json
{
  "buckets": [
    {
      "key": "engineering",
      "doc_count": 200,
      "level": 0,
      "education_level": {
        "buckets": [
          {
            "key": "bachelor",
            "doc_count": 120,
            "percentage": 60.0,
            "level": 1
          }
        ]
      },
      "children": [
        {
          "key": "development",
          "doc_count": 150,
          "level": 1,
          "parent_key": "engineering"
        }
      ]
    }
  ]
}
```

### 5.3 Range Analysis Response
```json
{
  "ranges": [
    {
      "key": "20-40",
      "from": 20,
      "to": 40,
      "doc_count": 300,
      "avg": 32.5,
      "groups": [
        {
          "key": "true",
          "doc_count": 60,
          "avg": 35.2
        },
        {
          "key": "false",
          "doc_count": 240,
          "avg": 31.8
        }
      ]
    }
  ]
}
```

---

## 7. Error Handling and Validation

### 7.1 Common Error Types
| Error Type | Cause | Solution |
|------------|-------|----------|
| Missing required field | Required parameter not provided | Check query structure |
| Invalid operator | Unsupported operator used | Use only supported operators |
| Type mismatch | Value type doesn't match field type | Ensure value type compatibility |
| Empty result | No documents match filters | Broaden filter criteria |
| Timeout | Query too complex or data too large | Add more filters, reduce fields |

### 7.2 Validation Rules
1. **Field existence**: Fields must exist in the index mapping
2. **Type compatibility**: Operators must be compatible with field types
3. **Range validity**: Range `from` must be less than `to` (if both specified)
4. **Array limits**: Maximum 10 fields, 3 group levels recommended
5. **Pagination**: Page must be ≥ 1, size must be ≤ 1000

---

## 8. Performance Considerations

### 8.1 Optimization Guidelines
1. **Indexing Strategy**:
   - Ensure numerical fields are indexed as appropriate types
   - Use keyword type for categorical fields used in group_by
   - Create composite indices for frequently queried combinations

2. **Query Design**:
   - Always include relevant filters to reduce dataset size
   - Use range queries instead of multiple OR conditions
   - Limit the number of aggregation levels
   - Set appropriate size limits for terms aggregations

3. **Resource Management**:
   - Monitor aggregation memory usage
   - Use pagination for large result sets
   - Consider time-based partitioning for time-series data

### 8.2 Recommended Limits
| Parameter | Recommended Limit | Hard Limit |
|-----------|------------------|------------|
| Fields per query | 5-10 | 20 |
| Group levels | 2-3 | 5 |
| Filters | 5-10 | 20 |
| Buckets per aggregation | 100-1000 | 10000 |
| Page size | 10-100 | 1000 |

---


## Summary

This enhanced JSON interface provides comprehensive statistical analysis capabilities for OpenSearch. Key features include:

1. **Complete Statistical Suite**: Support for all common statistical measures including Q1, Q3, Q5, median, mode, std deviation, and variance
2. **Flexible Range Analysis**: Custom range buckets for any numerical field (e.g., age groups, income brackets)
3. **Advanced Filtering**: 12 different filter operators including regex, wildcard, exists, and missing
4. **Hierarchical Aggregation**: Multi-level grouping with unlimited nesting
5. **Percentage Calculations**: Built-in percentage calculations for grouped data
6. **Pagination Support**: Built-in pagination for large result sets
7. **Type Safety**: Strong type checking and validation
8. **Performance Optimized**: Intelligent query construction with performance considerations

