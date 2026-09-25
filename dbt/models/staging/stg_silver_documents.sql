select
    doc_id, record_key, source, source_id, title, docket, record_type,
    cast(published_at as timestamp) as published_at,
    url, landing_url, n_pages, n_pages_total, n_chars, n_tables, parser,
    text_hash, duplicate_of, status, run_id,
    cast(parsed_at as timestamp)    as parsed_at
from {{ source('lake', 'silver_documents') }}
