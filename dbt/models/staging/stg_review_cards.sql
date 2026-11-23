select
    id as card_id,
    queue,
    species,
    suggested,
    model_label,
    truth,
    state,
    created_at,
    answered_at
from {{ source('petlens', 'review_card') }}
