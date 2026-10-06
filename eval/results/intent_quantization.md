# Intent model: fp32 vs dynamic int8

Measured in the Docker image (Linux, torch 2.14.1+cpu, one torch thread, 12 visible CPUs). Accuracy is in `intent_results.md` (rows `bert`, `bert_int8`).

| Variant | Weights | RSS after load (anon + file) | RSS after 155 queries (anon + file) | Peak RSS | Load time | Latency median | Latency p95 |
|---|---|---|---|---|---|---|---|
| fp32 | 417.7 MB | 390.6 MB (289.5 + 101.1) | 755.5 MB (303.0 + 452.6) | 755.6 MB | 3.19 s | 58.6 ms | 77.4 ms |
| int8 (dynamic, Linear layers) | 173.1 MB | 838.1 MB (412.3 + 425.8) | 869.0 MB (413.0 + 456.0) | 869.0 MB | 6.08 s | 18.5 ms | 25.2 ms |

RSS before loading the model (Python, pandas, torch imported): 252.1 MB.
