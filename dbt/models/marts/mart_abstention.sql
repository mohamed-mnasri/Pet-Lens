-- How often the policy answers, says "no clear breed" or sends to review: on each test
-- set of each evaluation, and on live uploads per day.
select
    'eval'::text as scope,
    eval_run_id,
    test_set::text as test_set,
    null::date as day,
    species::text as species,
    count(*)::bigint as n,
    avg((outcome = 'accept')::int)::double precision as accept_rate,
    avg((outcome = 'no_clear_breed')::int)::double precision as no_clear_breed_rate,
    avg((outcome = 'review')::int)::double precision as review_rate
from {{ ref('stg_eval_images') }}
group by 2, 3, 5
union all
select
    'live',
    null::uuid,
    null,
    created_day,
    species,
    count(*),
    avg((outcome = 'accept')::int),
    avg((outcome = 'no_clear_breed')::int),
    avg((outcome = 'review')::int)
from {{ ref('int_predictions') }}
group by 4, 5
