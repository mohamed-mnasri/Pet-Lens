-- Which breed gets taken for which, per evaluation and test set (known breeds only).
with images as (
    select * from {{ ref('stg_eval_images') }} where known_breed
),
per_label as (
    select eval_run_id, species, test_set, true_label, count(*) as n_label
    from images
    group by 1, 2, 3, 4
)
select
    i.eval_run_id,
    i.species::text as species,
    i.test_set::text as test_set,
    i.true_label::text as true_label,
    i.predicted_label::text as predicted_label,
    count(*)::bigint as n_confused,
    max(p.n_label)::bigint as n_label,
    (count(*)::double precision / max(p.n_label)) as share_of_label
from images as i
join per_label as p using (eval_run_id, species, test_set, true_label)
where not i.top1_correct
group by 1, 2, 3, 4, 5
