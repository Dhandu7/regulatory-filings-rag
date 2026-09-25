-- Every successfully parsed document must yield at least one chunk in the current chunk version
-- (all versions when the var isn't set).
with versions as (
    select distinct chunk_config_version from {{ ref('stg_silver_chunks') }}
    {% if var('chunk_config_version', none) %}
    where chunk_config_version = '{{ var("chunk_config_version") }}'
    {% endif %}
)
select d.doc_id, v.chunk_config_version
from {{ ref('stg_silver_documents') }} d
cross join versions v
left join {{ ref('stg_silver_chunks') }} c
  on c.doc_id = d.doc_id and c.chunk_config_version = v.chunk_config_version
where d.status = 'ok'
group by 1, 2
having count(c.chunk_id) = 0
