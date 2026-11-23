-- The VLM screener per model and prompt version: volume, parse failures, speed, tag mix.
select
    model_tag,
    prompt_version,
    max(model_digest) as model_digest,
    count(*)::bigint as photos,
    count(*) filter (where status = 'parse_failed')::bigint as parse_failed,
    count(*) filter (where attempts > 1)::bigint as retried,
    (percentile_cont(0.5) within group (order by seconds))::double precision as median_seconds,
    avg((kind <> 'photo')::int) filter (where status = 'ok')::double precision
        as share_not_photo,
    avg((not whole_animal)::int) filter (where status = 'ok')::double precision
        as share_partial,
    avg(several_animals::int) filter (where status = 'ok')::double precision as share_several,
    avg(blurry::int) filter (where status = 'ok')::double precision as share_blurry,
    avg(kitten::int) filter (where status = 'ok')::double precision as share_kitten
from {{ ref('stg_vlm_tags') }}
group by 1, 2
