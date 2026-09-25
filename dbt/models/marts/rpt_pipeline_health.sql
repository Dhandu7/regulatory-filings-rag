-- Per bronze run: what landed, what silver made of it, and the quarantine rate.
with landed as (
    select run_id, count(*) as landed, sum(bytes) as bytes, max(fetched_at) as fetched_at
    from {{ ref('stg_bronze_manifest') }} group by 1
),
outcome as (
    select m.run_id,
           count(distinct case when d.status = 'ok' then d.record_key end)        as parsed_ok,
           count(distinct case when d.status = 'duplicate' then d.record_key end) as duplicates,
           count(distinct q.record_key)                                          as quarantined
    from {{ ref('stg_bronze_manifest') }} m
    left join {{ ref('stg_silver_documents') }} d using (record_key)
    left join {{ ref('stg_silver_quarantine') }} q using (record_key)
    group by 1
)
select l.run_id, l.fetched_at, l.landed, l.bytes, o.parsed_ok, o.duplicates, o.quarantined,
       l.landed - o.parsed_ok - o.duplicates - o.quarantined as pending_silver,
       round(o.quarantined * 1.0 / nullif(l.landed, 0), 4)   as quarantine_rate
from landed l join outcome o using (run_id)
