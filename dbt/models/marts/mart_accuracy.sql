-- Headline metrics per evaluation, species and test set (one row each).
with m as (
    select * from {{ ref('stg_eval_metrics') }} where param is null
)
select
    r.eval_run_id,
    r.run_id,
    r.dataset_version,
    r.created_at as evaluated_at,
    m.species::text as species,
    m.test_set::text as test_set,
    max(m.n) filter (where m.metric in ('top1', 'coverage'))::bigint as n,
    max(m.value) filter (where m.metric = 'top1')::double precision as top1,
    max(m.lo) filter (where m.metric = 'top1')::double precision as top1_lo,
    max(m.hi) filter (where m.metric = 'top1')::double precision as top1_hi,
    max(m.value) filter (where m.metric = 'top3')::double precision as top3,
    max(m.value) filter (where m.metric = 'macro_f1')::double precision as macro_f1,
    max(m.value) filter (where m.metric = 'ece')::double precision as ece,
    max(m.value) filter (where m.metric = 'coverage')::double precision as coverage,
    max(m.value) filter (where m.metric = 'selective_accuracy')::double precision
        as selective_accuracy,
    max(m.value) filter (where m.metric = 'false_breed_claim_rate')::double precision
        as false_breed_claim_rate,
    max(m.value) filter (where m.metric = 'unknown_rejection_rate')::double precision
        as unknown_rejection_rate
from m
join {{ ref('stg_eval_runs') }} as r using (eval_run_id)
group by 1, 2, 3, 4, 5, 6
