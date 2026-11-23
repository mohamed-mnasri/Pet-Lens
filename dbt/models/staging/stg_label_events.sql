select
    id as label_event_id,
    image_id,
    detection_id,
    review_card_id,
    species,
    proposed_label,
    origin,
    action,
    review_ms,
    created_at
from {{ source('petlens', 'label_event') }}
