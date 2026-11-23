{{ config(unique_key='prediction_id', on_schema_change='append_new_columns') }}
-- Predictions from uploads, one row each, built incrementally by created_at.
select
    p.prediction_id,
    p.upload_id,
    u.kind as upload_kind,
    p.model_version,
    p.species,
    p.outcome,
    p.label,
    p.confidence,
    p.knn_distance,
    p.created_at,
    p.created_at::date as created_day
from {{ ref('stg_predictions') }} as p
join {{ ref('stg_uploads') }} as u using (upload_id)
{% if is_incremental() %}
where p.created_at > (select coalesce(max(created_at), '1970-01-01') from {{ this }})
{% endif %}
