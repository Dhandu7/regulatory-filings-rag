-- Near-duplicate detection: at most one 'ok' document per normalized text hash.
select text_hash, count(*) from {{ ref('stg_silver_documents') }}
where status = 'ok' group by 1 having count(*) > 1
