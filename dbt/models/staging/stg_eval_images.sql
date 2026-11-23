select
    eval_run_id,
    image_id,
    species,
    test_set,
    source,
    label_origin,
    true_label,
    known_breed,
    predicted_label,
    confidence,
    knn_distance,
    outcome,
    decided_label,
    top1_correct,
    flags
from {{ source('petlens', 'eval_image') }}
