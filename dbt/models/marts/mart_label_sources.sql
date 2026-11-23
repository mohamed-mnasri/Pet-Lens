-- Where labels come from in each dataset version, and how many reach training.
select
    dataset_version::text as dataset_version,
    source::text as source,
    label_origin::text as label_origin,
    sum(n)::bigint as images,
    (sum(n) filter (where split = 'train'))::bigint as in_train_split,
    (sum(n) filter (where split = 'excluded'))::bigint as excluded
from {{ ref('stg_dataset_counts') }}
group by 1, 2, 3
