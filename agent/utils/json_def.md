# OpenSearch Statistical Analysis JSON Interface Specification

## Overview

This document defines the JSON interface format for OpenSearch statistical analysis. This interface allows users to perform complex data statistical analysis through simple JSON configuration, including descriptive statistics, frequency analysis, and cross-analysis.

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
      "metrics": []
    }
  }
}
```

### Field Definitions
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query.type` | string | Yes | Query type, supported values in Section 2 |
| `query.config` | object | Yes | Query configuration object |
| `query.config.fields` | string[] | Yes | List of fields to analyze |
| `query.config.group_by` | string[] | No | List of grouping fields |
| `query.config.filters` | object[] | No | Array of filter conditions |
| `query.config.metrics` | string[] | No | Array of statistical metrics |

---

## 2. Query Types (query.type)

### 2.1 Available Types

| Type | Value | Description |
|------|-------|-------------|
| Descriptive Statistics | `"descriptive_stats"` | Statistical analysis for numerical fields (min, max, median, quartiles, etc.) |
| Frequency Analysis | `"frequency_analysis"` | Frequency analysis for categorical fields (count, proportion, etc.) |
| Cross Analysis | `"cross_analysis"` | Combined analysis supporting grouped statistics and cross-tabulation |

### 2.2 Type Selection Guide
- **Numerical field analysis** → Use `descriptive_stats`
- **Categorical field distribution** → Use `frequency_analysis`
- **Grouped comparative analysis** → Use `cross_analysis`

---

## 3. Configuration Parameters (query.config)

### 3.1 fields Parameter

#### Definition
```json
"fields": ["field1", "field2", ...]
```

#### Description
- **Type**: String array
- **Required**: Yes
- **Purpose**: Specifies field names to analyze
- **Constraints**: Must contain at least one field

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
| Wildcard match | `"like"` | Wildcard pattern matching | String | `"value": "张*"` |

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

| Metric | Value | Description | Applicable Types |
|--------|-------|-------------|------------------|
| Count | `"count"` | Number of documents | All types |
| Cardinality | `"cardinality"` | Number of unique values | Categorical fields |
| Minimum | `"min"` | Minimum value | Numerical fields |
| Maximum | `"max"` | Maximum value | Numerical fields |
| Average | `"avg"` | Average value | Numerical fields |
| Sum | `"sum"` | Sum of values | Numerical fields |
| Percentage | `"percentage"` | Percentage of total | Grouped statistics |
| Median | `"median"` | Median (50th percentile) | Numerical fields |
| First quartile | `"q1"` | First quartile (25th percentile) | Numerical fields |
| Third quartile | `"q3"` | Third quartile (75th percentile) | Numerical fields |
| Standard deviation | `"std_deviation"` | Standard deviation | Numerical fields |
| Variance | `"variance"` | Variance | Numerical fields |

#### Notes
- If `metrics` is not specified, the system automatically selects based on field type
- Numerical fields: Default includes `min`, `max`, `avg`, `median`, `q1`, `q3`
- Categorical fields: Default includes `count`, `cardinality`

---

## 4. Query Examples

### 4.1 Descriptive Statistics

**Function**: Analyze basic statistical information of numerical fields

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
      ],
      "metrics": ["min", "max", "avg", "median", "q1", "q3"]
    }
  }
}
```

**Output Field Description:**
- For each field, returns:
  - `count`: Number of documents
  - `min`: Minimum value
  - `max`: Maximum value
  - `avg`: Average value
  - `q1`: First quartile
  - `median`: Median
  - `q3`: Third quartile

### 4.2 Frequency Analysis

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

**Output Description:**
- Grouped by department
- Within each department, statistics for:
  - Distribution of education levels
  - Distribution of job titles
  - Includes count and percentage

### 4.3 Cross Analysis

**Function**: Complex grouped statistical analysis (e.g., disease proportions)

```json
{
  "query": {
    "type": "cross_analysis",
    "config": {
      "fields": ["has_disease", "smoking_status", "exercise_frequency"],
      "group_by": ["age_group", "gender"],
      "metrics": ["count", "percentage", "avg"],
      "filters": [
        {
          "field": "checkup_year",
          "operator": "gte",
          "value": 2022
        },
        {
          "field": "region",
          "operator": "in",
          "value": ["east", "south"]
        }
      ]
    }
  }
}
```

**Use Case Example:**
1. Among different age groups and genders:
   - Proportion with and without disease
   - Smoking status distribution
   - Average exercise frequency

---

## 5. Response Format

### 5.1 Descriptive Statistics Response Format
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
    "q3": 60.0
  },
  "income": {
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

### 5.2 Frequency Analysis Response Format
```json
{
  "buckets": [
    {
      "key": "engineering",
      "doc_count": 200,
      "education_level": {
        "buckets": [
          {
            "key": "bachelor",
            "doc_count": 120,
            "percentage": 60.0
          },
          {
            "key": "master",
            "doc_count": 60,
            "percentage": 30.0
          },
          {
            "key": "phd",
            "doc_count": 20,
            "percentage": 10.0
          }
        ]
      }
    }
  ]
}
```

### 5.3 Cross Analysis Response Format
```json
{
  "buckets": [
    {
      "key": "18-30",
      "doc_count": 300,
      "gender": {
        "buckets": [
          {
            "key": "male",
            "doc_count": 150,
            "has_disease": {
              "buckets": [
                {
                  "key": true,
                  "doc_count": 30,
                  "percentage": 20.0
                },
                {
                  "key": false,
                  "doc_count": 120,
                  "percentage": 80.0
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

---

## 6. Advanced Features

### 6.1 Compound Filter Conditions
Supports AND and OR logic:
```json
"filters": [
  {
    "bool": "and",  // or "or"
    "conditions": [
      {
        "field": "age",
        "operator": "gte",
        "value": 18
      },
      {
        "field": "age",
        "operator": "lte",
        "value": 60
      }
    ]
  }
]
```

### 6.2 Sorting Configuration
```json
"sort": [
  {
    "field": "doc_count",
    "order": "desc"  // "asc" or "desc"
  }
]
```

### 6.3 Pagination Support
```json
"pagination": {
  "page": 1,
  "size": 20
}
```

---

## 7. Important Notes

### 7.1 Performance Considerations
1. **Number of fields**: Recommended not to exceed 10 analysis fields per query
2. **Grouping levels**: Recommended not to exceed 3 grouping levels
3. **Data volume**: For large datasets, consider adding time range filters
4. **Pagination**: Use pagination for large result sets

### 7.2 Data Type Recommendations
| Analysis Type | Recommended Field Types | Not Recommended |
|---------------|------------------------|-----------------|
| Descriptive Statistics | Numerical (integer, float, date) | Text (non-numerical) |
| Frequency Analysis | Categorical fields (keyword, enum) | Long text (text) |
| Cross Analysis | Any type, but consider data distribution | High-cardinality fields |

### 7.3 Error Handling
- Non-existent field: Returns error message
- Type mismatch: Automatically skips or converts types
- Memory limits: Returns pagination suggestions
- Timeout: Suggests adding more filter conditions

### 7.4 Best Practices
1. **Define requirements clearly**: Determine analysis goals before designing queries
2. **Build incrementally**: Start with simple queries, gradually increase complexity
3. **Use filters**: Add filter conditions to reduce dataset size whenever possible
4. **Monitor performance**: Pay attention to query response time and resource usage
5. **Cache results**: Consider caching results for frequently executed queries

---

## Summary

This JSON interface provides powerful and flexible statistical analysis capabilities. Complex data analysis requirements can be implemented through simple configuration. The interface design follows these principles:

1. **Easy to use**: Define queries through intuitive JSON structure
2. **Flexible and extensible**: Supports multiple query types and operators
3. **Performance optimized**: Makes reasonable use of OpenSearch native features
4. **Type safe**: Clear data types and validation rules
5. **Well-documented**: Detailed error handling and best practices

Through this interface, users can:
- Quickly implement various statistical analysis requirements
- Avoid writing complex OpenSearch DSL
- Receive structured, easy-to-understand results
- Easily integrate with various data analysis applications