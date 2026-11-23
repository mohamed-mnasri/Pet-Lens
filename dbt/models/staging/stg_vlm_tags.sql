select
    id as vlm_tag_id,
    image_id::text as image_id,
    prompt_version::text as prompt_version,
    model_tag::text as model_tag,
    model_digest::text as model_digest,
    status::text as status,
    kind::text as kind,
    whole_animal,
    kitten,
    blurry,
    several_animals,
    attempts,
    seconds,
    created_at
from {{ source('petlens', 'vlm_tag') }}
