select
    source || ':' || source_id                       as record_key,
    source, source_id, title, url, docket, record_type, extension, content_type,
    cast(published_at as timestamp)                  as published_at,
    cast(registered_at as timestamp)                 as registered_at,
    cast(fetched_at as timestamp)                    as fetched_at,
    object_uri, sha256, bytes, size_hint, duplicate_of, run_id
from {{ source('lake', 'bronze_manifest') }}
