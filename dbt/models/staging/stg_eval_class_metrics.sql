select eval_run_id, species, test_set, label, n, recall, recall_lo, recall_hi, precision
from {{ source('petlens', 'eval_class_metric') }}
