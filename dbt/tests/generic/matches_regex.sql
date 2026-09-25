{% test matches_regex(model, column_name, pattern) %}
{#- DuckDB (local) and Spark SQL (Databricks) spell regex matching differently. -#}
select {{ column_name }} from {{ model }}
where {{ column_name }} is not null
{%- if target.type == 'duckdb' %}
  and not regexp_matches({{ column_name }}, '{{ pattern }}')
{%- else %}
  and not ({{ column_name }} rlike '{{ pattern }}')
{%- endif %}
{% endtest %}
