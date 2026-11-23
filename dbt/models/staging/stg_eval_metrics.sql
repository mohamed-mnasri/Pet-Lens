select eval_run_id, species, test_set, metric, param, value, lo, hi, n
from {{ source('petlens', 'eval_metric') }}
