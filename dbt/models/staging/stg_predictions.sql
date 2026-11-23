select
    id as prediction_id,
    upload_id,
    detection_id,
    group_idx,
    model_version,
    species,
    outcome,
    label,
    also,
    confidence,
    knn_distance,
    created_at
from {{ source('petlens', 'prediction') }}
