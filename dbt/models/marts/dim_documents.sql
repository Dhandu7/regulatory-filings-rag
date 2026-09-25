-- One row per unique, successfully parsed document, with its chunk counts per config version.
with chunks as (
    select doc_id, chunk_config_version, count(*) as n_chunks,
           sum(case when chunk_type = 'table' then 1 else 0 end) as n_table_chunks,
           sum(n_tokens) as n_tokens
    from {{ ref('stg_silver_chunks') }}
    group by 1, 2
)
select d.doc_id, d.source, d.source_id, d.title, d.docket, d.record_type, d.published_at, d.url,
       d.n_pages, d.n_tables, c.chunk_config_version, c.n_chunks, c.n_table_chunks, c.n_tokens
from {{ ref('stg_silver_documents') }} d
left join chunks c using (doc_id)
where d.status = 'ok'
