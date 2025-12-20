# OpenSearch Statistical Analysis JSON Interface Specification

## Overview

This document defines the JSON interface format for OpenSearch statistical analysis. Through unified JSON configuration, users can perform complex data statistical analysis, including numerical statistical calculations and multidimensional distribution analysis.

---

## 1. Basic Structure

### JSON Root Structure
```json
{
  "query": {
    "type": "string",        // Required: Query type
    "config": {              // Required: Query configuration
      "fields": [],
      "metrics": [],
      "dimensions": [],
      "groups": [],
      "buckets": [],
      "filters": []
    }
  }
}
```

### Field Definitions
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query.type` | string | Yes | Query type: `stats` (statistical calculation) or `distribution` (distribution analysis) |
| `query.config` | object | Yes | Query configuration object |
| `query.config.fields` | string[] | Conditional | Numerical fields to analyze (required for stats type) |
| `query.config.metrics` | string[] | Conditional | Statistical metrics to calculate |
| `query.config.dimensions` | string[] | Conditional | Dimension fields to analyze (required for distribution type) |
| `query.config.groups` | string[] | No | Grouping fields |
| `query.config.buckets` | object[] | No | Bucket configuration array |
| `query.config.filters` | object[] | No | Filter conditions array |

---

## 2. Query Types (query.type)

### 2.1 Available Types

| Type | Value | Description |
|------|-------|-------------|
| Statistical Calculation | `"stats"` | Statistical analysis of numerical fields |
| Distribution Analysis | `"distribution"` | Multidimensional distribution and grouping analysis |

### 2.2 Type Selection Guide
• Pure numerical statistics (averages, percentiles, standard deviation, etc.) → Use `stats`

• Distribution analysis (grouped statistics, frequency distribution, range bucketing, etc.) → Use `distribution`

---

## 3. Statistical Calculation Interface (stats)

### 3.1 Specific Configuration Parameters

```json
{
  "query": {
    "type": "stats",
    "config": {
      "fields": ["age", "salary", "score"],  // Required: Numerical field list
      "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation"], // Statistical metrics
      "filters": []  // Optional: Filter conditions
    }
  }
}
```

### 3.2 Supported Statistical Metrics (metrics)

| Metric | Value | Description |
|--------|-------|-------------|
| Count | `"count"` | Number of documents |
| Minimum | `"min"` | Minimum value |
| Maximum | `"max"` | Maximum value |
| Average | `"avg"` | Average value |
| Sum | `"sum"` | Sum of values |
| Median | `"median"` | Median (50th percentile) |
| First Quartile | `"q1"` | First quartile (25th percentile) |
| Third Quartile | `"q3"` | Third quartile (75th percentile) |
| Standard Deviation | `"std_deviation"` | Standard deviation |
| Variance | `"variance"` | Variance |

### 3.3 Default Metrics
• If `metrics` is not specified, defaults to: `["min", "max", "avg", "count"]`

---

## 4. Distribution Analysis Interface (distribution)

### 4.1 Specific Configuration Parameters

```json
{
  "query": {
    "type": "distribution",
    "config": {
      "dimensions": ["education", "job_title"],  // Required: Dimension fields to analyze
      "groups": ["department", "region"],        // Optional: Grouping fields
      "buckets": [                               // Optional: Bucket configuration
        {
          "type": "range",                       // Range bucket
          "field": "age",
          "ranges": [
            {"key": "Young", "from": 0, "to": 30},
            {"key": "Middle", "from": 30, "to": 60},
            {"key": "Old", "from": 60}
          ]
        },
        {
          "type": "terms",                       // Terms bucket
          "field": "education"
        }
      ],
      "metrics": ["count", "percentage", "avg"], // Metrics to calculate
      "metrics_field": "salary",                 // Optional: Target field for metric calculations
      "filters": []                              // Optional: Filter conditions
    }
  }
}
```

### 4.2 Bucket Types (buckets.type)

#### 4.2.1 Terms Bucket (terms)
For distribution analysis of categorical fields

```json
{
  "type": "terms",
  "field": "education_level",    // Field to bucket
  "size": 10                     // Optional: Number of buckets to return, default 10
}
```

#### 4.2.2 Range Bucket (range)
For range grouping of numerical fields

```json
{
  "type": "range",
  "field": "age",                // Numerical field to bucket
  "ranges": [                   // Range definitions
    {
      "key": "Young",           // Bucket identifier
      "from": 0,               // Start value (inclusive)
      "to": 30                 // End value (exclusive)
    },
    {
      "key": "Middle",
      "from": 30,
      "to": 60
    },
    {
      "key": "Old", 
      "from": 60               // Only from means >=60
    }
  ]
}
```

#### 4.2.3 Date Histogram Bucket (date_histogram)
For time-based bucketing analysis

```json
{
  "type": "date_histogram",
  "field": "create_time",       // Time field
  "interval": "1M",            // Time interval: 1d(day), 1w(week), 1M(month), 1y(year)
  "format": "yyyy-MM"          // Optional: Time format
}
```

### 4.3 Distribution Analysis Metrics (metrics)

| Metric | Value | Description | Notes |
|--------|-------|-------------|-------|
| Count | `"count"` | Document count | Always available |
| Percentage | `"percentage"` | Percentage within group | Requires grouping |
| Average | `"avg"` | Average value | Requires metrics_field |
| Sum | `"sum"` | Sum of values | Requires metrics_field |
| Minimum | `"min"` | Minimum value | Requires metrics_field |
| Maximum | `"max"` | Maximum value | Requires metrics_field |

### 4.4 Default Metrics
• If `metrics` is not specified, defaults to: `["count", "percentage"]`

---

## 5. Common Configuration Parameters

### 5.1 Filter Conditions (filters)

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

#### Supported Operators

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
| Exists | `"exists"` | Field exists (not null) | None | No value needed |

---

## 6. Query Examples

### 6.1 Statistical Calculation Examples

#### Basic Numerical Statistics
```json
{
  "query": {
    "type": "stats",
    "config": {
      "fields": ["age", "salary", "work_years"],
      "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation"],
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

#### Complete Statistical Calculation
```json
{
  "query": {
    "type": "stats", 
    "config": {
      "fields": ["blood_pressure", "cholesterol", "heart_rate"],
      "metrics": ["min", "max", "avg", "median", "q1", "q3", "std_deviation", "variance"],
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
```

### 6.2 Distribution Analysis Examples

#### Basic Frequency Distribution
```json
{
  "query": {
    "type": "distribution",
    "config": {
      "dimensions": ["education_level", "job_title"],
      "groups": ["department"],
      "metrics": ["count", "percentage"],
      "filters": [
        {
          "field": "active",
          "operator": "eq",
          "value": true
        }
      ]
    }
  }
}
```

#### Multi-level Grouping with Range Buckets
```json
{
  "query": {
    "type": "distribution",
    "config": {
      "dimensions": ["performance_rating", "attendance_rate"],
      "groups": ["company", "department"],
      "buckets": [
        {
          "type": "range",
          "field": "age",
          "ranges": [
            {"key": "20-30", "from": 20, "to": 30},
            {"key": "30-40", "from": 30, "to": 40},
            {"key": "40-50", "from": 40, "to": 50},
            {"key": "50+", "from": 50}
          ]
        },
        {
          "type": "range", 
          "field": "salary",
          "ranges": [
            {"key": "Low", "from": 0, "to": 10000},
            {"key": "Medium", "from": 10000, "to": 30000},
            {"key": "High", "from": 30000}
          ]
        }
      ],
      "metrics": ["count", "percentage", "avg"],
      "metrics_field": "salary"
    }
  }
}
```

#### Time Series Analysis
```json
{
  "query": {
    "type": "distribution", 
    "config": {
      "dimensions": ["product_category", "sales_region"],
      "buckets": [
        {
          "type": "date_histogram",
          "field": "sales_date",
          "interval": "1M",
          "format": "yyyy-MM"
        }
      ],
      "metrics": ["count", "sum", "avg"],
      "metrics_field": "sales_amount",
      "filters": [
        {
          "field": "sales_date",
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
```

#### Complex Multidimensional Analysis
```json
{
  "query": {
    "type": "distribution",
    "config": {
      "dimensions": ["has_disease", "treatment_type"],
      "groups": ["hospital", "department"],
      "buckets": [
        {
          "type": "range",
          "field": "age",
          "ranges": [
            {"key": "Child", "from": 0, "to": 12},
            {"key": "Teen", "from": 12, "to": 18},
            {"key": "Adult", "from": 18, "to": 60},
            {"key": "Senior", "from": 60}
          ]
        },
        {
          "type": "terms",
          "field": "gender",
          "size": 5
        }
      ],
      "metrics": ["count", "percentage", "avg", "max"],
      "metrics_field": "treatment_cost",
      "filters": [
        {
          "field": "visit_date",
          "operator": "gte",
          "value": "2023-01-01"
        }
      ]
    }
  }
}
```

---

## 7. Response Format

### 7.1 Statistical Calculation Response
```json
{
  "age": {
    "count": 1000,
    "min": 18,
    "max": 80,
    "avg": 45.5,
    "sum": 45500,
    "q1": 30.0,
    "median": 45.0,
    "q3": 60.0,
    "std_deviation": 15.2,
    "variance": 231.04
  },
  "salary": {
    "count": 1000,
    "min": 3000,
    "max": 50000,
    "avg": 15000.5,
    "sum": 15000500,
    "q1": 8000.0,
    "median": 12000.0,
    "q3": 20000.0
  }
}
```

### 7.2 Distribution Analysis Response
```json
{
  "buckets": [
    {
      "key": "20-30",
      "from": 20,
      "to": 30,
      "doc_count": 250,
      "metrics": {
        "count": 250,
        "percentage": 25.0,
        "avg_salary": 12000.5
      },
      "dimensions": {
        "education_level": {
          "buckets": [
            {
              "key": "Bachelor",
              "doc_count": 150,
              "percentage": 60.0,
              "avg_salary": 12500.0
            },
            {
              "key": "Master", 
              "doc_count": 100,
              "percentage": 40.0,
              "avg_salary": 14000.0
            }
          ]
        }
      },
      "groups": {
        "department": {
          "buckets": [
            {
              "key": "Engineering",
              "doc_count": 120,
              "percentage": 48.0
            }
          ]
        }
      }
    }
  ]
}
```

---

## 8. Error Handling and Validation

### 8.1 Common Error Types
| Error Type | Cause | Solution |
|------------|-------|----------|
| Missing required field | Required parameter not provided | Check query structure |
| Invalid operator | Unsupported operator used | Use only supported operators |
| Type mismatch | Value type doesn't match field type | Ensure value type compatibility |
| Empty result | No documents match filters | Broaden filter criteria |
| Timeout | Query too complex or data too large | Add more filters, reduce fields |

### 8.2 Validation Rules
1. Field existence: Fields must exist in the index mapping
2. Type compatibility: Operators must be compatible with field types
3. Range validity: Range `from` must be less than `to` (if both specified)
4. Array limits: Maximum 10 fields, 3 group levels recommended

---

## 9. Performance Considerations

### 9.1 Optimization Guidelines
1. Indexing Strategy:
   • Ensure numerical fields are indexed as appropriate types
   • Use keyword type for categorical fields used in grouping
   • Create composite indices for frequently queried combinations

2. Query Design:
   • Always include relevant filters to reduce dataset size
   • Use range queries instead of multiple OR conditions
   • Limit the number of aggregation levels
   • Set appropriate size limits for terms aggregations

3. Resource Management:
   • Monitor aggregation memory usage
   • Use pagination for large result sets
   • Consider time-based partitioning for time-series data

### 9.2 Recommended Limits
| Parameter | Recommended Limit | Hard Limit |
|-----------|------------------|------------|
| Fields per query | 5-10 | 20 |
| Group levels | 2-3 | 5 |
| Filters | 5-10 | 20 |
| Buckets per aggregation | 100-1000 | 10000 |