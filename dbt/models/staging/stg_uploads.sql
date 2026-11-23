select
    id as upload_id,
    kind,
    status,
    size_bytes,
    duration_s,
    created_at
from {{ source('petlens', 'upload') }}
