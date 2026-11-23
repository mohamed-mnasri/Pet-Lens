-- Swipe review per day: answers, answer mix, speed, and my accuracy on check cards.
select
    created_day as day,
    count(*)::bigint as answers,
    count(*) filter (where action = 'accept')::bigint as accepted,
    count(*) filter (where action = 'fix')::bigint as fixed,
    count(*) filter (where action = 'reject')::bigint as rejected,
    count(*) filter (where action = 'unsure')::bigint as unsure,
    count(*) filter (where action = 'skip')::bigint as skipped,
    (percentile_cont(0.5) within group (order by review_ms))::double precision
        as median_review_ms,
    count(*) filter (where queue = 'check' and action <> 'skip')::bigint as checks_answered,
    count(*) filter (
        where queue = 'check' and action in ('accept', 'fix') and proposed_label = truth
    )::bigint as checks_right
from {{ ref('int_label_events') }}
group by 1
