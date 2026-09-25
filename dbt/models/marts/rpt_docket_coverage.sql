select docket, count(distinct doc_id) as documents, min(published_at) as first_filing,
       max(published_at) as last_filing, sum(n_chunks) as chunks
from {{ ref('dim_documents') }}
where docket is not null
group by 1
