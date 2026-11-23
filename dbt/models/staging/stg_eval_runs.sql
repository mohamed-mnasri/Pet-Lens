select
    id as eval_run_id,
    run_id,
    dataset_version,
    embedding_model,
    created_at,
    row_number() over (partition by run_id order by created_at desc) = 1 as is_latest_for_run
from {{ source('petlens', 'eval_run') }}
