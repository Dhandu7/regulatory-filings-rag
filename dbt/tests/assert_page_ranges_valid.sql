select c.chunk_id
from {{ ref('stg_silver_chunks') }} c
join {{ ref('stg_silver_documents') }} d on d.doc_id = c.doc_id and d.status = 'ok'
where c.page_start < 1 or c.page_end < c.page_start or c.page_end > d.n_pages
