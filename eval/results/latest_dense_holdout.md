# Eval results: dense retrieval, holdout set

Index `v1-d4fd9d711f72__bge_small_en_v1_5`, 22 questions, k=6, neighbor expansion 0

| metric | value |
|---|---|
| doc_hit@1 | 0.591 |
| doc_hit@3 | 0.818 |
| doc_hit@6 | 0.864 |
| answer_hit@6 | 0.909 |
| mrr | 0.693 |
| answer_acc | 0.909 |
| answer_acc_judged | 0.909 |
| retrieval_ms_p50 | 18.5 |
| retrieval_ms_p95 | 27.2 |
| total_ms_p50 | 3208.4 |
| total_ms_p95 | 5574.9 |

| id | first relevant rank | answer evidence retrieved | retrieval ms |
|---|---|---|---|
| h01 | 3 | yes | 11 |
| h02 | 1 | yes | 14 |
| h03 | 1 | yes | 21 |
| h04 | 1 | yes | 19 |
| h05 | 1 | yes | 23 |
| h06 | 1 | yes | 21 |
| h07 | 1 | yes | 16 |
| h08 | 1 | yes | 19 |
| h09 | miss | yes | 18 |
| h10 | 1 | yes | 28 |
| h11 | 4 | yes | 19 |
| h12 | miss | no | 15 |
| h13 | 3 | yes | 27 |
| h14 | 2 | yes | 24 |
| h15 | 2 | yes | 22 |
| h16 | 3 | yes | 19 |
| h17 | miss | no | 17 |
| h18 | 1 | yes | 16 |
| h19 | 1 | yes | 16 |
| h20 | 1 | yes | 27 |
| h21 | 1 | yes | 16 |
| h22 | 1 | yes | 17 |
