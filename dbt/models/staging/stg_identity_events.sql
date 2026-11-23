select
    id as identity_event_id,
    pet_id,
    detection_id,
    other_pet_id,
    action::text as action,
    origin::text as origin,
    value::text as value,
    distance,
    created_at
from {{ source('petlens', 'identity_event') }}
