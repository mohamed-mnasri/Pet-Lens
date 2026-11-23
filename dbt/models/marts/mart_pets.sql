-- One row per pet that is still its own animal: photos, first and last seen, and how
-- many identity changes came from the matcher and how many from me.
with photos as (
    select
        pet_id,
        count(*)::bigint as photos,
        min(created_at) as first_seen,
        max(created_at) as last_seen
    from {{ ref('stg_detections') }}
    where pet_id is not null
    group by 1
),

changes as (
    select
        pet_id,
        count(*) filter (where origin = 'model')::bigint as model_events,
        count(*) filter (where origin = 'me')::bigint as my_events
    from {{ ref('stg_identity_events') }}
    group by 1
)

select
    p.pet_id,
    p.pet_name,
    p.species,
    p.status,
    coalesce(ph.photos, 0)::bigint as photos,
    ph.first_seen,
    ph.last_seen,
    coalesce(c.model_events, 0)::bigint as model_events,
    coalesce(c.my_events, 0)::bigint as my_events
from {{ ref('stg_pets') }} as p
left join photos as ph using (pet_id)
left join changes as c using (pet_id)
where p.status <> 'merged'
