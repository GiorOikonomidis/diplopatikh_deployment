# Pipeline rebuild log

## 1. Layer_1

- Model: `flat34_prune150density_noweights_64x64` — MLP [64,64], 8,034 params,
  flat 34-way classifier, raw features `tile_knn_25_OVN_09` (25 cols).
- Registered in MLflow Model Registry as
  `layer1_gate_flat34_prune150density_noweights_64x64` (v1).
  URI: `models:/layer1_gate_flat34_prune150density_noweights_64x64/latest`
- Evaluated live (real forward pass, loaded from the registry) in
  `try_modlels.py`, scored against `fine_label` on `real_test_annot_split3g.csv`.
- Result: **macro-F1 0.7696**, accuracy 0.9932.

## 2. Benign confidence-threshold sweep

- Script: `class_grouping/sweep_benign_threshold.py`, run on validation
  (`real_val_annot_split3g.csv`), Layer_1 live forward pass.
- Population: true Benign 27,519 / true soft-class 21,363 / Layer_1 predicted
  Benign 29,837 / Layer_1 predicted a soft class 19,052.

(corrected -- the original table's "Benign rows forwarded" column was
mislabeled: it counted rows Layer_1 *predicted* Benign with low confidence,
which mixes in other true labels, not an actual true-Benign subset.)

| tau | soft rows lost (pred=Benign, conf>=tau) | pred=Benign,conf<tau (any true label) | true-Benign rows kept | total forwarded |
|---|---|---|---|---|
| 0.900 | 265 (1.24%) | 7,887 | 5,834 / 27,519 | 27,197 |
| 0.950 | 115 (0.54%) | 10,466 | 8,263 / 27,519 | 29,626 |
| 0.970 | 60 (0.28%) | 12,204 | 9,946 / 27,519 | 31,309 |
| 0.990 | 14 (0.07%) | 15,497 | 13,193 / 27,519 | 34,556 |
| 0.995 | 6 (0.03%) | 17,468 | 15,156 / 27,519 | 36,519 |
| **0.999** | **2 (0.01%)** | **22,024** | **19,708 / 27,519** | **41,071** |

`total forwarded` and `true-Benign rows kept` now match
`build_flat34_layer2_dataset.py`'s actual val output (41,071 / 19,708) exactly.

- Chosen: **tau = 0.999**

## 3. Create layers_2 (Routers) dataset

Make new datasets that pass only the benigns that pass chossesn thershold and also add the last hidden embeddings of the layer_1 and relabeled via ipf_louvain

- Script: `class_grouping/build_flat34_layer2_dataset.py`
- Sources: `real_base150k_annot_split3g.csv` (train), `real_val_annot_split3g.csv`,
  `real_test_annot_split3g.csv`. Features: raw `tile_knn_25_OVN_09` + fresh
  Layer_1 h0 (live forward pass through the registered model).
- Output: `datasets/layer2_{train,val,test}_flat34grounded.csv`

| split | rows in | kept | BenignTraffic | G1 | G3 | G2 |
|---|---|---|---|---|---|---|
| train | 2,178,226 | 191,400 | 92,287 | 75,473 | 22,074 | 1,566 |
| val | 1,176,851 | 41,071 | 19,708 | 16,236 | 4,776 | 351 |
| test | 1,176,851 | 41,177 | 19,923 | 16,194 | 4,716 | 344 |

All soft-class counts (G1/G3/G2) match the full true-instance totals exactly
-- every soft row kept regardless of confidence, as intended. Not yet
trained on.

## 4. Train layers_2 (Routers) 
go back to the original batch/lr , 5e10-4 / 512
EXP_NAME : FLAT34_Layer2 , all other settings as usual  instead of the ones below
- model_diems : [128] , [64,64]
- model_input : features + hidden , features , hidden 
- data prep : 
  - stage_1 : no / cap to the median 
  - stage_2 : no / oversample 10x , 
  - stage_3 : no / yes  
- weights: no/full

Rules for the data-prep stages (not yet built):
- weights always computed from the PRE-preprocessing population counts,
  even once stage1/2/3 change what's actually trained on (reuse
  `class_weight_source_csv` in run.py, point it at the flat34grounded CSV).
- preprocessing acts on raw features only, ignores h0. Downsample (stage1)
  and Tomek (stage3) just select subsets of real rows -- their existing h0
  stays valid. SMOTE (stage2) invents new rows via raw-feature
  interpolation -- their h0 must come from a REAL Layer_1 forward pass on
  the synthetic raw vector afterward, never interpolated (h0 is Layer_1's
  nonlinear function of raw features, not a linear one).

Round 1 (no prep stages, baseline): 12 runs, lr=5e-4/batch=512, experiment
`FLAT34_Layer2`. Ran to completion.

| run | feature_set | hidden | weights | val F1 | test F1 | test acc |
|---|---|---|---|---|---|---|
| **featplus_128_wnone** | features+hidden | [128] | none | **0.8385** | **0.8275** | 0.9061 |
| featplus_6464_wnone | features+hidden | [64,64] | none | 0.8384 | 0.8269 | 0.9053 |
| hiddenonly_6464_wnone | hidden only | [64,64] | none | 0.8374 | 0.8257 | 0.9057 |
| hiddenonly_128_wnone | hidden only | [128] | none | 0.8355 | 0.8240 | 0.9053 |
| featplus_6464_wfull | features+hidden | [64,64] | full | 0.7217 | 0.7164 | 0.8749 |
| hiddenonly_128_wfull | hidden only | [128] | full | 0.7200 | 0.7120 | 0.8680 |
| featplus_128_wfull | features+hidden | [128] | full | 0.7179 | 0.7103 | 0.8708 |
| hiddenonly_6464_wfull | hidden only | [64,64] | full | 0.7153 | 0.7100 | 0.8696 |
| featonly_6464_wnone | features only | [64,64] | none | 0.6176 | 0.6029 | 0.8142 |
| featonly_128_wnone | features only | [128] | none | 0.5982 | 0.5824 | 0.8038 |
| featonly_6464_wfull | features only | [64,64] | full | 0.5730 | 0.5676 | 0.7338 |
| featonly_128_wfull | features only | [128] | full | 0.5573 | 0.5521 | 0.7128 |

Winner by raw score: `flat34grounded_featplus_128_wnone` -- test macro-F1
**0.8275**, acc 0.9061. Same pattern as every earlier round: unweighted
beats weighted decisively; h0 matters a lot (`featonly` tops out at 0.60);
width barely matters.

**Chosen for simplicity: `flat34grounded_hiddenonly_6464_wnone`** -- test
macro-F1 **0.8257**, acc 0.9057. Essentially tied with the raw winner
(-0.18%) but input is just the 64-d Layer_1 hidden state alone, no raw
features carried alongside -- simpler pipeline for negligible cost. This is
the router going forward.

Registered in the MLflow Model Registry as **`layer_2_hiddenonly_6464_wnone`**.
URI: `models:/layer_2_hiddenonly_6464_wnone/latest`

## 5. Test Layer_1 + layers_2 (Routers) 
Now we need a proof of concept of so far . We want to connect all together so far the layer_1 , layer_2 and the exit threashold . Also layer 2 we dont want it to work with the hidden from the input data , rather gathering in inference from the layer_1 , we want full classification results , original and clustered . So we pass the whole test dataset from the pipeline not the parts that each one layer want and aggregate and estimation

- Script: `class_grouping/eval_layer1_layer2_poc.py`. Whole test set
  (`real_test_annot_split3g.csv`, all 1,176,851 rows), no pre-filtering by
  ground truth -- gate decision comes from Layer_1's OWN prediction:
  predicted hard class -> always exit; predicted Benign -> exit if
  conf>=0.999 else forward; predicted a soft class -> always forward.
  Layer_2 gets fed h0 from Layer_1's live forward pass on this same run,
  never a precomputed column.
- Gate: 1,135,678/1,176,851 (96.5%) exit at Layer_1 -- 1,127,890 hard-class,
  7,788 confident-Benign. 41,173 forwarded to Layer_2.

**CLUSTERED** (24 groups: 20 hard classes as singletons + Benign + G1/G2/G3):
macro-F1 **0.9648**, acc 0.9959. G1 f1=0.9117, G3 f1=0.8034, G2 f1=0.6616
(weak, only 344 support).

| group | precision | recall | f1-score | support |
|---|---|---|---|---|
| BenignTraffic | 0.9279 | 0.9651 | 0.9461 | 27,709 |
| DDoS-ACK_Fragmentation | 0.9911 | 0.9947 | 0.9929 | 7,292 |
| DDoS-HTTP_Flood | 0.9688 | 0.9647 | 0.9668 | 709 |
| DDoS-ICMP_Flood | 0.9999 | 0.9998 | 0.9999 | 180,447 |
| DDoS-ICMP_Fragmentation | 0.9957 | 0.9962 | 0.9960 | 11,402 |
| DDoS-PSHACK_Flood | 0.9997 | 0.9997 | 0.9997 | 103,326 |
| DDoS-RSTFINFlood | 0.9998 | 0.9996 | 0.9997 | 101,819 |
| DDoS-SYN_Flood | 0.9994 | 0.9990 | 0.9992 | 102,208 |
| DDoS-SlowLoris | 0.9285 | 0.9598 | 0.9439 | 622 |
| DDoS-SynonymousIP_Flood | 0.9995 | 0.9987 | 0.9991 | 90,480 |
| DDoS-TCP_Flood | 0.9993 | 0.9998 | 0.9995 | 113,735 |
| DDoS-UDP_Flood | 0.9995 | 0.9994 | 0.9994 | 136,717 |
| DDoS-UDP_Fragmentation | 0.9938 | 0.9931 | 0.9934 | 7,224 |
| DoS-HTTP_Flood | 0.9922 | 0.9867 | 0.9894 | 1,805 |
| DoS-SYN_Flood | 0.9967 | 0.9993 | 0.9980 | 51,116 |
| DoS-TCP_Flood | 0.9999 | 0.9990 | 0.9994 | 67,697 |
| DoS-UDP_Flood | 0.9993 | 0.9992 | 0.9993 | 83,627 |
| **G1** | 0.9184 | 0.9052 | 0.9117 | 16,194 |
| **G2** | 0.9665 | 0.5029 | 0.6616 | 344 |
| **G3** | 0.8751 | 0.7426 | 0.8034 | 4,716 |
| Mirai-greeth_flood | 0.9988 | 0.9989 | 0.9989 | 24,934 |
| Mirai-greip_flood | 0.9985 | 0.9987 | 0.9986 | 19,279 |
| Mirai-udpplain | 0.9996 | 0.9996 | 0.9996 | 22,536 |
| VulnerabilityScan | 0.9636 | 0.9562 | 0.9599 | 913 |
| **accuracy** | | | **0.9959** | 1,176,851 |
| **macro avg** | 0.9796 | 0.9566 | **0.9648** | 1,176,851 |
| weighted avg | 0.9959 | 0.9959 | 0.9958 | 1,176,851 |

**ORIGINAL** (per-fine-class, forwarded rows scored as ceiling -- reached
the right group, not yet a resolved fine class since no experts wired in):
weakest same as every round -- Uploading_Attack 0.333, XSS 0.417,
Backdoor_Malware 0.562, CommandInjection 0.580, SqlInjection 0.662.
Everything else above 0.87, most above 0.99.

Overall row-level correctness (exact for exited, ceiling for forwarded):
**0.9959**.

Logged to MLflow: experiment `LAYER1_LAYER2_POC`, run
`layer1_gate_layer2_hiddenonly`. First genuinely connected, live,
no-oracle-shortcut pipeline number this session.

## 6. Train experts
mostlik;y we will go to the the 



## 7. Runtime cost: torch/LightGBM vs ONNX Runtime

Same weights, same inputs, two runtimes: the models as loaded natively
(PyTorch router + LightGBM experts) against the same models exported to ONNX
and run under ONNX Runtime. Measured per model in isolation, so the numbers
are the models' own cost -- no pandas, no test CSV resident.

### Setup

Intel Core i5-3570 @ 3.40 GHz (4 cores, 1 thread/core), 7 GB RAM,
Linux 7.1.8-arch1-3. Python 3.12.14, PyTorch 2.6.0, LightGBM 4.7.0,
ONNX Runtime 1.29.0, NumPy 2.4.4, scikit-learn 1.8.0.

Single-flow inference (batch = 1), 25 raw features (`tile_knn_25_OVN_09`),
single-threaded on both sides (`intra_op_num_threads=1` /
`torch.set_num_threads(1)`) so the CPU comparison is like-for-like. Same
weights both paths: the 64x64 MLP router (7,384 params, 24-way) plus the three
LightGBM experts. Feature scaling is folded into the router in both paths
(`ScaledRouter` for the export, the same `(x - mean) / std` applied natively),
so neither side does less work than the other.

Code: `metrics/model_benchmark.py` (ONNX), `metrics/model_benchmark_native.py`
(torch/LightGBM), sharing `metrics/hardware_consumption.py` and
`metrics/energy_consumption.py`.

### How each metric was taken

| metric | what it is | how |
|---|---|---|
| on-disk size | serialized model bytes | `os.path.getsize()`; runtime-independent, so this is the one to quote as "model size" |
| marginal RAM | RSS the model's weights cost | `psutil` RSS around session/model construction, in a **fresh subprocess**, with a *different* model loaded and run first so the runtime's one-time init is not charged to the model |
| framework import | cost of the runtime itself | `psutil` RSS delta across `import torch` / `import lightgbm` / `import onnxruntime`, before any model exists |
| wall-clock | real time per forward pass | `time.perf_counter()` over 5,000 passes after 200 warm-up passes |
| CPU time | CPU-seconds consumed (user + sys) | `psutil.Process().cpu_times()` delta / iterations -- separates compute from waiting |
| CPU utilization | core saturation | `psutil.Process().cpu_percent()`, primed at block entry; 100% = one core saturated |
| RSS growth | leak check | RSS delta across the 5,000-pass loop; ~0 means nothing accumulates per call |
| energy / CO2 | power attributed to the process | `codecarbon`, in a **separate pass** auto-scaled to >=30 s |
| macro-F1 | composed accuracy, 34 classes | full 1,176,851-row test set, router + expert dispatch |

Two measurement traps worth recording, both found the hard way:

- **The energy tracker cannot share a loop with the latency measurement.**
  Wrapping the same tight loop in codecarbon inflated router latency from
  0.021 ms to 0.95 ms and dropped measured CPU from ~103% to ~11% -- its
  polling thread contends for the GIL. A long run (tens of seconds) dilutes
  this to nothing, but any short benchmark is wrecked by it. Latency and
  energy are now measured in separate passes.
- **Warm-up is mandatory, and the first session in a process is not
  representative.** ONNX Runtime allocates its arena and thread pool lazily on
  the first `InferenceSession`: 9.6 MB for session 1, then 3.9 / 0.9 / 0.16 MB
  for sessions 2-4. Charging that to whichever model loads first overstates it
  by ~9.6 MB.

### Per-model results

| | router (MLP) | | expert G1 | | expert G3 | | expert G2 | |
|---|---|---|---|---|---|---|---|---|
| | native | ONNX | native | ONNX | native | ONNX | native | ONNX |
| wall-clock (ms) | 0.0976 | **0.0205** | 0.0677 | **0.0354** | 0.0609 | **0.0290** | 0.0577 | **0.0266** |
| CPU time (ms) | 0.0980 | **0.0200** | 0.0660 | **0.0340** | 0.0620 | **0.0300** | 0.0560 | **0.0260** |
| marginal RAM (MB) | 3.273 | **0.066** | 17.396 | **3.772** | 16.605 | **1.077** | 16.540 | **1.217** |
| energy (kWh/inf) | 5.10e-10 | **1.12e-10** | 3.51e-10 | **1.87e-10** | 3.18e-10 | **1.53e-10** | 3.02e-10 | **1.40e-10** |
| speed-up | -- | **4.8x** | -- | **1.9x** | -- | **2.1x** | -- | **2.2x** |

CPU time equals wall-clock and system time is 0.000 ms in every case: the
models are pure userspace compute, no I/O and no syscall overhead. RSS growth
is ~0.000 MB across all four, so nothing accumulates per inference.

### System totals

| | native (torch + LightGBM) | ONNX Runtime | ratio |
|---|---|---|---|
| framework import | 374.6 MB (torch) + 154.0 MB (lightgbm) | **19.4 MB** | **27x** |
| runtime one-time init | included above | 9.6 MB | -- |
| model weights in RAM | 53.8 MB | **6.1 MB** | **8.8x** |
| **total footprint** | **~450 MB** | **~35 MB** | **~13x** |
| on-disk model size | 1.35 MB | **0.76 MB** | 1.8x |
| latency per flow | 0.0988 ms | **0.0211 ms** | **4.7x** |
| throughput, 1 core | ~10,100 flows/s | **~47,400 flows/s** | 4.7x |
| composed macro-F1 | **0.8606639** | **0.8603587** | **-0.000305** |

Latency per flow weights the router (every row) against the experts (the
1.86% of rows that reach one: 21,925 / 1,176,851).

Where the two runtimes win is not the same place. The router is where ONNX
wins on speed (4.8x) and RAM (50x: 0.066 MB vs 3.273 MB) -- it is a 7,384-param
MLP, and torch's per-call Python/dispatch overhead dominates a model that
small. The experts are where ONNX wins on RAM in absolute terms: a LightGBM
Booster resident in memory is 16.5-17.4 MB against 1.1-3.8 MB for the same
trees as ONNX, despite the serialized files being within ~10% of each other.

The headline for the paper: the export costs 0.0003 macro-F1 and buys 4.7x
throughput at ~1/13 the memory footprint.

### Open / unverified

- **Conversion did NOT preserve predictions exactly.** macro-F1 0.8606639
  (torch) vs 0.8603587 (ONNX), a real -0.000305. Inference is deterministic
  -- same weights, same rows, same order, no randomness -- so this is not
  run-to-run noise: some rows genuinely classify differently after conversion.
  Most likely cause is the LightGBM->ONNX tree conversion, which stores split
  thresholds in float32 while LightGBM compares in float64, so rows sitting
  near a threshold fall the other way. The router (a dense MLP) is far less
  exposed to this than the tree ensembles. Negligible at 3e-4 macro-F1, but it
  must be disclosed rather than described as "identical" -- and quantified:
  dump both runs' label arrays and diff elementwise for the actual number of
  changed rows and which classes they land in. Not yet done.
- The native macro-F1 (0.8606639) comes from `try_modlels.py`, which carries a
  G1-vs-G3 override branch the ONNX script does not have. That branch was
  separately measured to have zero net effect (identical 0.8607 with and
  without), so the comparison holds, but a clean native pass with identical
  dispatch logic would remove the objection.
- Energy is whole-process draw over the window / N, so it includes process
  baseline, not marginal model cost. Subtracting an idle-loop baseline would
  be needed for strictly marginal energy.
- The marginal-RAM method likely understates each model slightly, since the
  warm-up session's allocator may already hold headroom pages the measured
  session reuses without growing RSS. Treat ~6 MB (ONNX) as a lower bound;
  the honest range is 6-15 MB depending on whether runtime init is charged to
  the models.
- Single run each, no repetitions, so no variance estimate on the timing and
  energy figures (the macro-F1 gap is exempt: inference is deterministic, so
  it needs no repetitions to be real).
