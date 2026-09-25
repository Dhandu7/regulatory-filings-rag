-- Chunks of the current chunking version must be non-trivial and within the configured ceiling
-- (tables allow up to max_table_tokens). Older versions are immutable history and aren't re-tested.
select chunk_id, chunk_config_version, n_tokens
from {{ ref('stg_silver_chunks') }}
where (n_tokens <= 0 or n_tokens > {{ var('max_chunk_tokens') }})
{% if var('chunk_config_version', none) %}
  and chunk_config_version = '{{ var("chunk_config_version") }}'
{% endif %}
