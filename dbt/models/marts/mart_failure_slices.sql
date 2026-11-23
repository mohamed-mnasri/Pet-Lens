-- Error rate by slice: photo source, label origin, true breed and photo flag.
with images as (
    select * from {{ ref('stg_eval_images') }} where known_breed
),
slices as (
    select eval_run_id, species, test_set, 'source' as slice_type, source as slice_value,
        top1_correct from images
    union all
    select eval_run_id, species, test_set, 'label_origin', label_origin, top1_correct from images
    union all
    select eval_run_id, species, test_set, 'breed', true_label, top1_correct from images
    union all
    select eval_run_id, species, test_set, 'flag', f.flag, top1_correct
    from images, jsonb_array_elements_text(images.flags) as f(flag)
)
select
    eval_run_id,
    species::text as species,
    test_set::text as test_set,
    slice_type::text as slice_type,
    slice_value::text as slice_value,
    count(*)::bigint as n,
    count(*) filter (where not top1_correct)::bigint as errors,
    (count(*) filter (where not top1_correct))::double precision / count(*) as error_rate
from slices
group by 1, 2, 3, 4, 5
