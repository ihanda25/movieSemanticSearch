# Cross-encoder fine-tune comparison

Production `cross-encoder/ms-marco-MiniLM-L-6-v2` vs. `models/cross-encoder-finetuned-v1/final`, scored on the same already-retrieved candidates per query -- see `evaluation/finetune_cross_encoder.py` and `findings.txt`'s 2026-09-17 entry for how the checkpoint was produced. Full per-query rows: `evaluation/finetune_comparison.csv`.

## Aggregate metrics (MRR@5 / NDCG@5)

| Split | n | Base (bi-encoder) | Production reranker | Fine-tuned reranker |
|---|---:|---:|---:|---:|
| train | 264 | 0.599 / 0.648 | 0.673 / 0.710 | **0.730 / 0.764** |
| val | 57 | 0.573 / 0.627 | 0.699 / 0.732 | **0.731 / 0.755** |
| test | 56 | 0.629 / 0.673 | 0.704 / 0.743 | **0.753 / 0.793** |

## Biggest improvements (35 queries moved up)

| Split | Movie | Before rank | After rank |
|---|---|---:|---:|
| train | Bambi (1942) | 5 | 1 |
| train | Up (2009) | 3 | 1 |
| train | The Muppet Christmas Carol (1992) | 3 | 1 |
| test | The Princess and the Frog (2009) | 4 | 2 |
| test | Alice in Wonderland (1951) | 3 | 1 |
| train | Toy Story (1995) | 5 | 4 |
| train | Incredibles 2 (2018) | 3 | 2 |
| train | Cars 2 (2011) | 2 | 1 |
| train | Brave (2012) | 4 | 3 |
| train | Inside Out (2015) | 2 | 1 |
| train | Captain America: Civil War (2016) | 2 | 1 |
| train | Peter Pan (1953) | 2 | 1 |
| train | One Hundred and One Dalmatians (1961) | 3 | 2 |
| train | The Parent Trap (1998) | 2 | 1 |
| train | A Goofy Movie (1995) | 2 | 1 |
| train | Snow White and the Seven Dwarfs (1938) | 2 | 1 |
| train | 20,000 Leagues Under the Sea (1954) | 2 | 1 |
| train | I'll Be Home for Christmas (1998) | 2 | 1 |
| train | Tuck Everlasting (2002) | 2 | 1 |
| train | Star Wars: Episode III - Revenge of the Sith (2005) | 3 | 2 |

## Regressions (4 queries moved down)

| Split | Movie | Before rank | After rank |
|---|---|---:|---:|
| train | McFarland, USA (2015) | 2 | 4 |
| train | Finding Dory (2016) | 2 | 3 |
| train | Thor: Ragnarok (2017) | 1 | 2 |
| val | Tarzan (1999) | 2 | 3 |

## Not among the 5 stored candidates (77 queries)

This only checks the 5 candidates already stored from collection time, so it cannot tell a true stage-1 ceiling miss (confirmed movie beyond the bi-encoder's full 50, unreachable by any reranker) apart from a stage-2 demotion (movie WAS in the 50, a reranker pushed it below rank 5, and a better reranker could in principle recover it). See `evaluation/diagnose_stage1_ceiling.py`, which re-retrieves the full 50 and separates the two -- run it for the current breakdown; `evaluation/stage1_ceiling.csv` has the up-to-date numbers.

