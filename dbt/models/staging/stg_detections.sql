select
    d.id as detection_id,
    f.upload_id,
    d.species::text as species,
    d.group_idx,
    d.pet_id,
    d.created_at
from {{ source('petlens', 'detection') }} as d
join {{ source('petlens', 'frame') }} as f on f.id = d.frame_id
