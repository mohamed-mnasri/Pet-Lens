select
    id as pet_id,
    species::text as species,
    name::text as pet_name,
    status::text as status,
    merged_into,
    created_at
from {{ source('petlens', 'pet') }}
