select c.*, d.record_type
from {{ ref('stg_silver_chunks') }} c
join {{ ref('stg_silver_documents') }} d on d.doc_id = c.doc_id and d.status = 'ok'
