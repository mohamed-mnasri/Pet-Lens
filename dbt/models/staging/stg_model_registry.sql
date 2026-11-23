select
    id as registry_id,
    run_id::text as run_id,
    dataset_version::text as dataset_version,
    weights_hash::text as weights_hash,
    config ->> 'embedding_model' as embedding_model,
    champion as is_champion,
    created_at as registered_at,
    promoted_at,
    (gate ->> 'passed')::boolean as gate_passed,
    (gate #>> '{numbers,golden set,diff}')::double precision as gate_golden_gain,
    (gate #>> '{numbers,unseen-source test,diff}')::double precision as gate_unseen_gain
from {{ source('petlens', 'model_registry') }}
