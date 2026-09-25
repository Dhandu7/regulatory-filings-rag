select
    chunk_id, doc_id, chunk_config_version, ordinal, chunk_type, section,
    page_start, page_end, n_tokens, text, context, source, source_id, docket, title,
    cast(published_at as timestamp) as published_at, url, run_id,
    cast(chunked_at as timestamp)   as chunked_at
from {{ source('lake', 'silver_chunks') }}
-- re-runs of the same config append identical deterministic ids; keep the latest copy
qualify row_number() over (partition by chunk_config_version, chunk_id order by chunked_at desc) = 1
