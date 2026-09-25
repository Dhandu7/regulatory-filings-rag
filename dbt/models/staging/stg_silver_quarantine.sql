select record_key, source, source_id, sha256, object_uri, stage, reason, run_id,
       cast(quarantined_at as timestamp) as quarantined_at
from {{ source('lake', 'silver_quarantine') }}
