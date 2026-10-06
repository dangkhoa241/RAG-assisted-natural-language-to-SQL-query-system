# Intent model: fp32 vs dynamic int8

Measured in Docker (Linux), one inference thread, 12 visible CPUs. Accuracy is in `intent_results.md` (rows `bert`, `bert_int8`, `bert_onnx_int8`).

| Variant | Weights | RSS after load (anon + file) | RSS after 155 queries (anon + file) | Peak RSS | Load time | Latency median | Latency p95 |
|---|---|---|---|---|---|---|---|
| fp32 (torch) | 417.7 MB | 389.0 MB (289.4 + 99.6) | 754.4 MB (303.0 + 451.4) | 754.4 MB | 4.7 s | 60.9 ms | 76.0 ms |
| int8 (torch, dynamic, Linear layers) | 173.1 MB | 838.5 MB (412.6 + 425.9) | 869.0 MB (413.1 + 455.9) | 869.0 MB | 7.62 s | 19.0 ms | 25.5 ms |
| int8 ONNX (onnxruntime) | 105.1 MB | 239.5 MB (193.5 + 46.0) | 242.8 MB (193.8 + 49.0) | 242.8 MB | 1.41 s | 12.7 ms | 15.6 ms |

RSS before loading the model (Python and pandas, plus torch for the torch variants): fp32 (torch) 250.8 MB, int8 (torch, dynamic, Linear layers) 251.9 MB, int8 ONNX (onnxruntime) 67.7 MB.
