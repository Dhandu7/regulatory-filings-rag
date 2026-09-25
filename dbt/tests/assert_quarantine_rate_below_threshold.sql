{{ config(severity = 'warn') }}
-- A spike in quarantined files usually means the parser or the source changed.
select run_id, quarantine_rate
from {{ ref('rpt_pipeline_health') }}
where quarantine_rate > {{ var('max_quarantine_rate') }}
