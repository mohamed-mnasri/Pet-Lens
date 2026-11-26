# VLM screener: qwen3-vl:4b, prompt screen-v1

- Photos tagged: 1500 (valid JSON: 1491, parse_failed: 9, needed the retry: 20)
- Median time per photo: 7.1 s
- Kinds: photo 1462, drawing 22, other 6, toy 1; partial 16%, several 11%, blurry 3%, kitten 17%

Agreement (Cohen's kappa: 1 is perfect, 0 is chance):

| Comparison | Photos | Kappa | Both say yes | Only the VLM | Only the other |
| --- | ---: | ---: | ---: | ---: | ---: |
| not a real photo vs my 'not a cat or dog' | 96 | 0.00 | 0 | 1 | 0 |
| unusable vs my 'not sure' | 96 | 0.00 | 0 | 36 | 0 |
| blurry vs the pixel check | 1491 | 0.23 | 7 | 41 | 3 |
| whole animal vs the box check | 595 | 0.02 | 264 | 205 | 67 |

Effect on label noise (web photos I reviewed, 'not sure' left out):

- Wrong label among photos the VLM keeps: 2% of 60
- Wrong label among photos the VLM would drop: 0% of 36
- Web training candidates it keeps out of training: 378 of 1491 tagged (25%)
