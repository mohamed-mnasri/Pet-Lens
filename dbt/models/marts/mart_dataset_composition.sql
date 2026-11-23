-- Images per breed and split in each dataset version, with the share of its species.
with per_breed as (
    select dataset_version, species, breed, split, sum(n) as n
    from {{ ref('stg_dataset_counts') }}
    group by 1, 2, 3, 4
)
select
    dataset_version::text as dataset_version,
    species::text as species,
    breed::text as breed,
    split::text as split,
    n::bigint as n,
    (n::double precision / sum(n) over (partition by dataset_version, species, split))
        as share_of_species_split
from per_breed
