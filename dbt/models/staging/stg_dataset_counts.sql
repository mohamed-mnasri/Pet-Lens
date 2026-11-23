select dataset_version, source, species, breed, split, label_origin, n
from {{ source('petlens', 'dataset_count') }}
