{{ config(unique_key='label_event_id', on_schema_change='append_new_columns') }}
-- Review answers with their card's queue, built incrementally (label events are
-- append-only, so new ids only ever arrive).
select
    e.label_event_id,
    e.review_card_id,
    c.queue,
    e.species,
    e.proposed_label,
    c.suggested,
    c.truth,
    e.origin,
    e.action,
    e.review_ms,
    e.created_at,
    e.created_at::date as created_day
from {{ ref('stg_label_events') }} as e
left join {{ ref('stg_review_cards') }} as c on c.card_id = e.review_card_id
{% if is_incremental() %}
where e.label_event_id > (select coalesce(max(label_event_id), 0) from {{ this }})
{% endif %}
