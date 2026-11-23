-- One row per registered run: the champion flag, the last gate, and the headline numbers
-- from the run's latest evaluation (pooled over species, weighted by photos).
with latest_eval as (
    select a.*
    from {{ ref('mart_accuracy') }} as a
    join {{ ref('stg_eval_runs') }} as r using (eval_run_id)
    where r.is_latest_for_run
),

pooled as (
    select
        run_id,
        max(evaluated_at) as last_evaluated_at,
        sum(top1 * n) filter (where test_set = 'golden')
            / nullif(sum(n) filter (where test_set = 'golden' and top1 is not null), 0)
            as golden_top1,
        sum(top1 * n) filter (where test_set = 'test_unseen_source')
            / nullif(sum(n) filter (where test_set = 'test_unseen_source' and top1 is not null), 0)
            as unseen_top1,
        max(false_breed_claim_rate) filter (where test_set = 'test_domestic')
            as false_breed_claim_rate
    from latest_eval
    group by 1
)

select
    g.run_id,
    g.dataset_version,
    g.embedding_model,
    g.weights_hash,
    g.is_champion,
    g.registered_at,
    g.promoted_at,
    g.gate_passed,
    g.gate_golden_gain,
    g.gate_unseen_gain,
    p.last_evaluated_at,
    p.golden_top1::double precision as golden_top1,
    p.unseen_top1::double precision as unseen_top1,
    p.false_breed_claim_rate::double precision as false_breed_claim_rate
from {{ ref('stg_model_registry') }} as g
left join pooled as p using (run_id)
