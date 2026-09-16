# Keyword/tagline enrichment comparison

Same 50 premise queries, embedding model and cross-encoder; candidate_k=50.
Original baseline.json and results.csv preserved.

| Metric | Baseline | Enriched |
|---|---:|---:|
| Top 1 | 0.700 | 0.820 |
| Top 5 | 0.840 | 0.960 |
| MRR | 0.762 | 0.878 |
| Stage-1 Recall@50 | 0.940 | 0.980 |

## Changed ranks

| Movie | Before | After |
|---|---:|---:|
| Toy Story (1995) | 5 | 4 |
| Inside Out (2015) | 2 | 1 |
| Guardians of the Galaxy (2014) | outside candidates | 20 |
| Tarzan (1999) | 5 | 2 |
| Peter Pan (1953) | 7 | 2 |
| Up (2009) | outside candidates | 4 |
| Fantasia (1940) | 19 | 2 |
| The Empire Strikes Back (1980) | 15 | 3 |
| The Avengers (2012) | 9 | 1 |
| The Incredibles (2004) | 2 | 1 |
| Bambi (1942) | 5 | 1 |
| Thor: Ragnarok (2017) | 8 | 1 |
| Rogue One: A Star Wars Story (2016) | 2 | 1 |

## Remaining misses

- Guardians of the Galaxy (2014): bi rank 29, rerank 20; first result Chicken Little.
- Star Wars (1977): bi rank 64, rerank None; first result Tall Tale: The Unbelievable Adventure.
