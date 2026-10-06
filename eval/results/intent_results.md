# Intent classifier evaluation

## Accuracy / macro-F1

| Model | val (n=1000) | hard (n=150) | in_domain_hard (n=75) | unseen_domain (n=75) |
|---|---|---|---|---|
| keyword | 0.657 / 0.669 | 0.240 / 0.136 | 0.240 / 0.133 | 0.240 / 0.139 |
| tfidf_lr | 0.999 / 0.999 | 0.693 / 0.689 | 0.667 / 0.662 | 0.720 / 0.718 |
| bert | 1.000 / 1.000 | 0.847 / 0.841 | 0.853 / 0.837 | 0.840 / 0.844 |
| bert_int8 | 1.000 / 1.000 | 0.833 / 0.828 | 0.827 / 0.818 | 0.840 / 0.839 |
| bert_onnx_int8 | 1.000 / 1.000 | 0.840 / 0.834 | 0.853 / 0.841 | 0.827 / 0.829 |

Cells are accuracy / macro-F1.

## Per-class F1 on the hard set

| Model | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| keyword | 0.10 | 0.00 | 0.21 | 0.38 | 0.00 |
| tfidf_lr | 0.69 | 0.77 | 0.55 | 0.67 | 0.77 |
| bert | 0.90 | 0.95 | 0.64 | 0.80 | 0.92 |
| bert_int8 | 0.86 | 0.91 | 0.60 | 0.80 | 0.97 |
| bert_onnx_int8 | 0.92 | 0.92 | 0.64 | 0.80 | 0.90 |

## Confusion matrices on the hard set (rows = true, cols = predicted)

**keyword**

| true \ pred | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| aggregate | 2 | 0 | 2 | 26 | 0 |
| compare | 1 | 0 | 1 | 28 | 0 |
| count | 6 | 0 | 4 | 20 | 0 |
| filter | 0 | 0 | 0 | 30 | 0 |
| trend | 2 | 0 | 2 | 26 | 0 |

**tfidf_lr**

| true \ pred | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| aggregate | 22 | 3 | 1 | 1 | 3 |
| compare | 2 | 25 | 0 | 3 | 0 |
| count | 7 | 2 | 14 | 6 | 1 |
| filter | 0 | 5 | 3 | 21 | 1 |
| trend | 3 | 0 | 3 | 2 | 22 |

**bert**

| true \ pred | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| aggregate | 27 | 0 | 1 | 1 | 1 |
| compare | 0 | 27 | 0 | 3 | 0 |
| count | 3 | 0 | 15 | 10 | 2 |
| filter | 0 | 0 | 0 | 30 | 0 |
| trend | 0 | 0 | 1 | 1 | 28 |

**bert_int8**

| true \ pred | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| aggregate | 25 | 0 | 5 | 0 | 0 |
| compare | 0 | 26 | 0 | 4 | 0 |
| count | 3 | 0 | 15 | 11 | 1 |
| filter | 0 | 0 | 0 | 30 | 0 |
| trend | 0 | 1 | 0 | 0 | 29 |

**bert_onnx_int8**

| true \ pred | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| aggregate | 28 | 0 | 1 | 1 | 0 |
| compare | 0 | 27 | 0 | 3 | 0 |
| count | 3 | 0 | 15 | 10 | 2 |
| filter | 0 | 0 | 0 | 30 | 0 |
| trend | 0 | 2 | 1 | 1 | 26 |

