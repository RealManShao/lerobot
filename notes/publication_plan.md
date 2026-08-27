# Publication Plan: DrifOv, Overlap-Conditioned One-Step Action Generation

## Problem and proposed direction

The work ports the training objective from *Implicit Drifting Policy* (IDP, arXiv:2606.01098v1) into the GR00T N1.7 vision-language-action stack. Drifting replaces GR00T's flow-matching action head with a deterministic one-step transformer while retaining the GR00T visual-language backbone and preprocessing. DrifOv extends that action head with overlap conditioning so that it can preserve an unexecuted action prefix and generate its successor suffix in one evaluation.

For a main robotics conference, this is not yet publishable as a new method. The implementation is principally an architectural port of an existing objective, and the available evidence is a preliminary synchronous LIBERO comparison plus offline model-forward timing rather than a statistically supported asynchronous-control result. The revised work should explicitly call IDP an inherited training objective and make the new scientific contribution:

> An overlap-conditioned one-step VLA action head that accepts the unexecuted prefix of the current action chunk and predicts a continuous successor chunk in one network evaluation, enabling asynchronous RTC-like chunk transitions without iterative denoising.

The completed LIBERO study isolates the action-head change: Drifting, DrifOv, and GR00T N1.7 were trained on the same suite data and evaluated with the recorded synchronous protocol. The remaining evaluation is real-robot validation, where asynchronous overlap-conditioned generation is part of the method rather than merely a deployment optimization. All policies will run on the same inference server, but each should be allowed to operate at its own maximum stable policy rate. The low-level controller interpolates joint-angle commands, so policy update rate and actuator servo rate must still be reported separately.

## Completed LIBERO evidence: action-head-only comparison

The four available `main/latency_bench/libero_*/summary.json` files are the current source of record. They compare the three action heads on the same suite-specific training data. They establish synchronous policy quality and model-forward timing; they do **not** exercise DrifOv's asynchronous overlap path, nor do they provide seed variation, confidence intervals, raw latency distributions, or end-to-end control-loop latency.

| LIBERO suite | Drifting | DrifOv | GR00T N1.7 |
| --- | ---: | ---: | ---: |
| Spatial | 66% | **74%** | 66% |
| Object | **78%** | 72% | 80% |
| Goal | 74% | **80%** | 72% |
| LIBERO-10 | **48%** | 46% | 36% |
| Macro average | 66.5% | **68.0%** | 63.5% |

Across suites, the measured mean action-head time is approximately 4.8--5.0 ms for Drifting and DrifOv, versus 45.3--45.9 ms for GR00T N1.7. The corresponding measured backbone-plus-head sum is approximately 29.2--30.2 ms versus 69.9--70.9 ms. These are CUDA-synchronized model-forward components, not wall-clock deployment latency. The DrifOv--Drifting macro-average difference is 1.5 percentage points; without repeated training/evaluation seeds, it is preliminary rather than evidence of a reliable quality gain.

## Reviewer-style assessment of the current work

### Likely verdict today: weak reject / reject

#### Strengths

- The implementation preserves the core IDP training structure:
  - zero-seed proposal at deployment time;
  - expert-proximal noisy probe at `t=0.9`;
  - geometry-weighted proposal and proximal potential losses;
  - detached observation geometry;
  - row-standardized similarities and softmax neighborhoods;
  - per-minibatch reference variance;
  - deterministic one-evaluation action-head inference.
- The GR00T integration is practically relevant because it tests IDP-style training in a large VLA setting rather than the smaller policy backbones used by the source paper.
- The latency profile separates preprocessing, VLM, action head, and queue effects, correctly showing that the VLM dominates model latency.
- The implementation and presentation explicitly distinguish Drifting's inherited one-step objective from DrifOv's learned overlap conditioning and hard prefix preservation.
- The completed LIBERO comparison provides an initial four-suite action-head-only screen, and the collected soft-bag dataset provides a strong deformable-object task where stale actions and slow replanning can visibly affect alignment quality.

#### Major rejection risks

1. **Insufficient novelty in the current version.** Replacing the GR00T action generator with a transformer trained using the published IDP objective is an implementation contribution. The revised paper must demonstrate that overlap-conditioned one-step generation is technically distinct from IDP and materially improves continuity or responsive control.
2. **Policy-performance evidence is preliminary.** The completed four-suite LIBERO action-head comparison reports synchronous success rates (DrifOv 68.0%, Drifting 66.5%, GR00T N1.7 63.5% macro average), but has no repeated seeds, confidence intervals, raw rollouts, asynchronous-overlap evaluation, or external pi0.5 comparison. Real-robot evidence remains pending.
3. **Confounded comparison.** The current work simultaneously changes the action architecture, inference process, and training objective. Any gain could come from the new 12-layer transformer, alternating image/text cross-attention, state dropout, parameter count, or optimization rather than conditional expert geometry.
4. **Latency scope is limited.** The completed summaries include GR00T N1.7 and separate backbone/action-head timing, but only report means for CUDA-synchronized model-forward components. They do not establish matched wall-clock latency, tail latency, memory, or live-loop performance.
5. **Weak statistics.** Thirty repeated warm-cache measurements from three fixed frames cannot support a reliable P99 claim. Repetition of fixed inputs also does not measure input-dependent variance.
6. **No live-loop evidence.** The offline benchmark omits bridge networking, camera/state synchronization, action transmission, low-level interpolation, command age, jitter, and real control-loop deadline misses.
7. **The overlap mechanism is not yet validated in rollout.** DrifOv can consume action prefixes and preserves them exactly, but the completed synchronous LIBERO protocol does not provide a non-empty prefix. It therefore cannot establish queue-stall prevention or compare fairly with pi0.5's real-time chunking behavior.
8. **Geometry remains minibatch-dependent.** Both local and reference geometry vary with batch composition. This is inherited from IDP and should be disclosed; task/embodiment conditioning is a secondary extension only if mixed-data experiments require it.
9. **Method-fidelity ambiguity.** The source paper's main equation excludes the query sample from local covariance, while its hyperparameter table says the full minibatch includes `j=i`. The current implementation includes self-similarity. This choice must be made explicit and ablated.
10. **Representation choice is unvalidated.** The implementation builds the geometry embedding from pooled projected VLM tokens plus encoded state. The source paper uses the final observation-encoder representation before the task head. The adaptation is reasonable, but its effect must be isolated.
11. **Deterministic mode collapse remains.** Zero-seed one-step inference cannot represent balanced multimodal action distributions without an additional latent or mode mechanism. Claims must be limited accordingly.
12. **No reproducible publication artifact.** The manual benchmark contains workspace-specific paths, and the current documentation is user-facing rather than a frozen experimental protocol with seeds, dataset splits, checkpoints, and raw result files.

## Paper-to-code correspondence to establish before new experiments

Create a formal method audit covering:

- IDP Eq. 6: proposal plus weighted proximal objective -> current `forward`.
- IDP Eqs. 7-10: detached normalized features, row z-score, conditional variance, normalized precision, reference ratio, ReLU excess -> current `compute_geometry_excess`.
- IDP Algorithm 1: zero proposal and proximal noisy anchor -> current `_predict` calls.
- IDP deployment rule `f(o, 0, 0)` -> current `get_action`.
- Explicit adaptation choices not in IDP:
  - GR00T/Cosmos-Qwen backbone;
  - pooled VLM-plus-state geometry embedding;
  - embodiment-specific encoders and decoder;
  - alternating image/text cross-attention;
  - state dropout;
  - action masks and padded heterogeneous dimensions;
  - action chunk size 40;
  - frozen versus trainable backbone components.

Add equation-level unit tests with hand-computed tensors. Test masked coordinates, mixed action dimensions, batch size one, duplicate observations, zero-variance coordinates, numerical stability, and both include-self/exclude-self variants.

## Proposed method: overlap-conditioned one-step action head

Use a working name such as **Overlap-Conditioned Drifting (OC-Drift)** or **One-Step Overlap Policy**. Do not use "Implicit Drifting Policy" as the new method name.

### 1. Prefix-aware action representation

Extend the action head input with:

- the aligned unexecuted prefix from the current action chunk;
- a binary prefix-valid mask for every action timestep and dimension;
- an overlap-length or execution-offset embedding;
- optional command-age or planned-start-time conditioning.

When no prefix is supplied, the head reduces to the existing zero-seed one-step proposal. When a prefix is supplied, the transformer attends to the known action segment and predicts only the continuation. Preserve valid prefix values exactly in the returned chunk and execute only the newly predicted suffix.

This changes the action head and inference interface, rather than only changing the IDP loss.

### 2. Overlap-conditioned training

Construct training examples from temporally overlapping demonstration windows:

- sample a replan stride and overlap length, including zero-overlap examples;
- align the earlier chunk's remaining actions with the target successor chunk;
- train the suffix under the inherited IDP potential while masking fixed prefix coordinates from the prediction loss;
- add a boundary objective on action, velocity, and optionally acceleration at the prefix/suffix transition;
- corrupt some training prefixes with bounded noise, temporal offsets, or detached model predictions to reduce expert-prefix exposure bias;
- log performance separately for clean expert prefixes and model-generated prefixes.

Start with expert prefixes plus bounded corruption. Add an EMA/self-generated prefix buffer only if train-test mismatch remains measurable; do not introduce it speculatively.

### 3. One-evaluation asynchronous overlap generation

During rollout:

- trigger inference before the current chunk is exhausted;
- pass the still-unexecuted actions as the prefix;
- associate every request with observation time, intended execution time, and queue state;
- reject or shorten stale results under a documented policy;
- append only the generated suffix after the preserved overlap;
- perform exactly one action-head evaluation for each successor chunk.

This is RTC-like continuity through learned prefix conditioning, but it is not iterative RTC guidance. Use precise terminology and compare the mechanisms directly.

### 4. Optional conditioned geometry

Task/embodiment-compatible neighborhoods, cross-batch memory, and hierarchical reference geometry remain a secondary extension. Implement them only after the prefix head is validated, and include them in the main paper only if they add independent LIBERO gains. This avoids a kitchen-sink method and keeps the novelty centered on the action head.

## Implementation snapshot (2026-08-17)

The overlap-conditioned method is implemented as the separate `drif_ov` policy. The original `drifting` policy remains unchanged as the inherited, unconditioned IDP-port baseline.

Implemented:

- explicit action-prefix and per-coordinate prefix-valid-mask inputs;
- learned prefix-mask, overlap-length, and execution-offset embeddings;
- one-evaluation inference with exact hard preservation of valid prefix values;
- zero-prefix fallback to unconditioned one-step generation;
- suffix-only proposal and proximal potentials;
- configurable inherited geometry weighting on the suffix or full chunk;
- expert-prefix sampling from aligned demonstration action windows, including zero-overlap examples;
- bounded expert-prefix corruption;
- configurable action, velocity, and acceleration boundary objectives;
- overlap, corruption, suffix-horizon, input-prefix-error, hard-preservation-error, and boundary diagnostics;
- asynchronous LeRobot RTC-engine integration using unexecuted leftovers, explicit pre-padding prefix length, generic relative-action re-anchoring, overlap-expiry rejection, and delay-aware queue replacement;
- checkpointed switches for endpoint-only, proximal-only, geometry, prefix, corruption, embedding, and boundary-loss ablations.

Deferred rather than claimed as complete:

- a separate earlier-window dataset field or model-generated/EMA prefix replay buffer;
- command-age and planned-start timestamp conditioning beyond the current discrete execution offset;
- native GR00T relative-action prefix re-anchoring (the unsafe combination is rejected);
- request IDs, configurable stale-result rejection thresholds, and raw live-loop event logging;
- publication benchmark scripts and the complete matched-baseline experiment matrix.

## Implementation workstreams

### A. Correctness and configurability

- [x] Add prefix tensor, prefix-valid mask, overlap length, and execution offset to the action-head interface.
- [x] Implement hard prefix preservation, suffix-only loss masking, and boundary continuity losses.
- [x] Sample aligned expert prefixes within each episode-safe demonstration action window.
- [ ] Add an explicit earlier-window dataset field or model-generated prefix replay if bounded corruption is insufficient.
- [ ] Add request IDs, planned-start timestamps, request-age logging, and configurable stale-result rejection.
- [x] Preserve masks throughout neighbor covariance and reference statistics.
- [x] Add diagnostic outputs for overlap length, prefix source/corruption, input and hard-preservation prefix errors, boundary action/velocity/acceleration jump, execution offset, and suffix horizon.
- [x] Make implemented publication variants selectable from checkpointed configuration.

### A.1 Completed simulation comparison

- [x] Train and evaluate Drifting, DrifOv, and GR00T N1.7 on the same training dataset for each of LIBERO Spatial, Object, Goal, and LIBERO-10.
- [x] Record synchronous success and separated mean backbone/action-head model-forward times in suite `summary.json` files.
- [ ] Export raw rollout outcomes, seeds, checkpoints, and per-call timing samples so the current summary-only result can support uncertainty estimates and reproducibility.

### B. Matched baselines

Implement baselines within the same GR00T backbone and preprocessing stack:

1. GR00T N1.7 flow-matching head at the standard inference schedule.
2. GR00T flow head at reduced NFE values where supported.
3. Same Drifting transformer with plain endpoint MSE.
4. Same transformer with proximal loss but no geometry.
5. Current unconditioned IDP port.
6. Current IDP port plus asynchronous prefetch but no prefix conditioning.
7. Current IDP port plus heuristic blending.
8. Proposed prefix-conditioned head with endpoint MSE only.
9. Proposed prefix-conditioned head with proximal loss but no geometry.
10. Full proposed prefix-conditioned head with inherited IDP geometry.

Keep trainable backbone modules, optimizer, data, augmentation, batch size, training updates, checkpoint selection, action horizon, parameter count, and compute budget matched. Report deviations explicitly.

Add two external large-policy baselines on the same real-robot dataset:

11. pi0.5 with its recommended real-time chunking configuration.
12. pi0.5 without RTC or with a controlled common execution rule, if supported.

Use both controlled rate/delay tables and a native-speed best-system table. A direct comparison between Drifting without RTC and pi0.5 with RTC is valid as an equal-server system comparison, but it must not by itself be used to attribute gains solely to the action generator.

GPU-memory comparisons must distinguish:

- full-policy peak allocated and reserved memory on the common server;
- steady-state inference memory and cache state;
- action-head-only parameters and activation memory;
- same-backbone Drifting versus GR00T, which supports causal attribution to the head;
- Drifting versus pi0.5, which is a whole-system comparison across different model families and sizes.

### C. Reproducible measurement

- Replace hard-coded local benchmark paths with CLI/config inputs.
- Save raw per-sample latency and rollout events, not only summary statistics.
- Record package versions, GPU, model revision, dataset revision, checkpoint hash, warmup, precision, compilation mode, and random seeds.
- Use CUDA events for isolated GPU kernels and synchronized wall-clock timing for end-to-end latency.
- Separate cold start, steady-state offline, and live-robot measurements.
- Freeze the inference server, GPU allocation, power/performance mode, bridge, camera inputs, precision, compilation settings, and competing processes across methods.
- Define maximum stable policy rate before evaluation, for example as the highest requested rate that remains below a fixed deadline-miss fraction and command-age bound over a long calibration rollout.

## Experimental program

### Research questions

- **RQ1:** Does overlap-conditioned one-step generation improve success and chunk-boundary continuity over the current IDP port, asynchronous prefetch, and heuristic blending?
- **RQ2:** Does learned prefix conditioning retain one action-head evaluation while approaching or exceeding pi0.5 RTC and GR00T chunk continuity?
- **RQ3:** Which training components are necessary: overlapping windows, prefix corruption, offset/timing embeddings, IDP geometry, and boundary losses?
- **RQ4:** What accuracy-latency-memory-update-rate trade-off does one-step generation provide relative to GR00T flow matching and pi0.5 RTC on the same server?
- **RQ5:** Does asynchronous prefetch translate lower action-head cost into fewer live control deadline misses without harming action continuity?
- **RQ6:** On deformable-object shaking and alignment, does Drifting's higher sustainable update rate and lower command age improve grasp retention, alignment accuracy, disturbance recovery, and overall task success relative to pi0.5 RTC and GR00T DiT?

### LIBERO simulated evaluation

The initial synchronous comparison is complete and summarized above. Treat it as an action-head-only screening result, not a final asynchronous-control claim. The next LIBERO iteration should:

- retain the completed Spatial, Object, Goal, and LIBERO-10 suites and official task definitions;
- freeze the current suite-specific data, preprocessing, action horizon, precision, checkpoints, and evaluation protocol in an experiment manifest;
- at least three training seeds;
- fixed, preregistered evaluation initial states or at least 50 episodes per task/seed;
- report mean, standard deviation, bootstrap 95% confidence intervals, and paired tests where initial states are shared.

Evaluate both zero-prefix rollouts and asynchronous overlapping rollouts. LIBERO should establish that prefix conditioning does not sacrifice standard benchmark success and should measure action discontinuity even when the environment itself is not highly dynamic.

Include an overlap stress study:

- vary overlap length and replan stride;
- vary prefix corruption and model-prefix error;
- inject observation and inference delay;
- test stale-result rejection thresholds;
- measure success, suffix prediction error, boundary discontinuity, and effective command age.

### Physical-robot soft-bag evaluation (reserved for ongoing engineering work)

Reserve this section until robot-control engineering is stable. Do not report or imply real-robot results before the asynchronous inference, state/camera synchronization, queue replacement, and safety behavior operate smoothly. Once ready, make the collected grasp-shake-align task the central physical-robot experiment. Define the task as reproducible phases:

1. grasp a deformable soft bag from randomized initial configurations;
2. shake or manipulate it without losing the grasp;
3. align a defined bag feature, opening, edge, or pose to a target;
4. hold or place it within a measurable tolerance.

Compare Drifting, GR00T N1.7 DiT, and pi0.5 trained on the same demonstrations and deployed on the same server. Include the current package-to-box task as a secondary conventional manipulation result if its data and protocol are already available.

Use the same low-level joint-angle controller and interpolation for every method, but let each policy run at its measured maximum stable update rate in the primary native-speed comparison. Also run shared-rate and delay-matched controls. Log requested rate, achieved rate, policy outputs, interpolated low-level trajectories, and observation-to-command age. For each method:

- use at least three independently trained seeds when feasible;
- use randomized bag shape/configuration, grasp point, target orientation, lighting/background, and controlled perturbations;
- evaluate enough trials for confidence intervals (target 30-50 trials per task/method, not 10);
- blind or automate success scoring;
- report phase success and end-to-end success;
- report alignment error, time to align, grasp-loss rate, recovery success, command age, action jerk, chunk-boundary discontinuity, and deadline misses;
- report failures by category: perception, failed grasp, bag slip, insufficient shaking, overshoot, alignment failure, trajectory/manifold violation, chunk-boundary discontinuity, timeout, and unsafe action rejection.

Do not describe the low-level interpolated motion as evidence that the VLA itself runs at the actuator servo frequency. The defensible claim is that, on identical server hardware, lower-latency one-step inference can sustain a higher policy update rate and lower command age, which may improve deformable-object task performance while the shared low-level controller ensures smooth joint motion.

### Ablations

Minimum ablation matrix:

- endpoint only;
- + proximal evaluation;
- + unconditioned local geometry;
- + prefix-conditioned overlapping windows;
- clean expert prefixes versus corrupted prefixes;
- fixed versus variable overlap lengths;
- prefix-valid mask and offset embedding on/off;
- suffix-only geometry versus geometry on the full chunk;
- action boundary loss, velocity loss, and acceleration loss;
- asynchronous prefetch without conditioning;
- heuristic blending;
- optional model-generated/EMA prefix training;
- include-self versus exclude-self;
- VLM-only, state-only, and fused geometry embeddings;
- synchronous versus asynchronous rollout;
- pi0.5 RTC on/off and GR00T chunk-continuity on/off where supported;
- Drifting asynchronous prefetch and blending on/off;
- a safe policy-rate sweep spanning rates supported by all methods plus each method's native maximum;
- artificial-delay injection into Drifting to match the slower baselines' latency or command-age distributions.

### Efficiency and systems metrics

Report:

- action-head NFE;
- action-head and full-model parameter counts;
- FLOPs or a consistent compute proxy;
- training time and peak memory;
- steady-state and peak GPU allocated/reserved memory on the same server;
- P50/P95/P99 latency with at least hundreds to thousands of diverse observations;
- full-pipeline throughput and control-loop deadline-miss rate;
- requested versus achieved policy update rate and maximum stable rate;
- live bridge receive, preprocessing, VLM, head, postprocessing, queue wait, and action-send times;
- stale-observation age and chunk-boundary action discontinuity;
- low-level interpolated trajectory jerk and tracking error;
- success versus wall-clock latency, not latency alone.

The central efficiency claim must be precise: the method reduces action-head evaluations and action-head latency. End-to-end speed and memory claims require same-server measurements. The current profile shows the VLM is about 76% of median model time, so a large head-level speedup may become only a modest full-policy speedup. The real-robot claim must be supported by mediation evidence: rate sweeps and delay injection should show that lower command age or a higher achieved update rate improves bag-task outcomes.

## Paper framing

### Defensible contribution statement

1. Identify the missing capability in existing one-step IDP-style policies: they cannot condition successor chunks on an already executing action prefix.
2. Introduce an overlap-conditioned action head and training procedure that predicts a continuous suffix in one network evaluation.
3. Show preliminary four-suite LIBERO evidence that changing to DrifOv's overlap-aware action head retains competitive synchronous success while preserving the one-step head's low model-forward cost.
4. Validate responsive control on a deformable soft-bag task only after the real-robot system is stable, then compare pi0.5 RTC and GR00T DiT on the same server with latency, GPU memory, achieved update rate, command age, chunk continuity, and task performance.

### Claims to avoid

- Do not claim invention of IDP, conditional expert geometry, proximal evaluation, or one-step drifting.
- State explicitly that IDP supplies the base potential objective; the new contribution is prefix-aware architecture, overlapping-window training, and one-evaluation successor-chunk generation.
- Do not call the current model-forward component means end-to-end or live-control latency.
- Do not claim real-time control from action-head NFE alone.
- Do not call low-level interpolation a high-frequency VLA policy.
- Do not attribute a native-speed system difference solely to the action head when pi0.5 RTC, GR00T chunking, model size, and Drifting scheduling differ; use same-backbone and delay/rate controls for attribution.
- Do not claim multimodal generation with deterministic zero-seed inference.
- Do not generalize from one Siemens task or one checkpoint.

### Suggested paper structure

1. Problem: one-step VLA generation lacks native overlap-aware chunk continuity.
2. Related work: IDP, one-step policy acceleration, RTC, action chunking, streaming/asynchronous policy inference.
3. Analysis: why prefetch or blending alone cannot make the successor prediction depend on the executing prefix.
4. Method: prefix-aware head, overlapping-window training, boundary objectives, and asynchronous one-step generation.
5. Experimental protocol and matched baselines.
6. Completed LIBERO synchronous success and model-forward component means; then repeated-seed overlap stress tests and continuity results.
7. Reserved: soft-bag quality, responsiveness, RTC/chunking, and live control-loop results after robot engineering is complete.
8. Ablations, limitations, safety, and reproducibility.

## Publication gates

Do not submit until all of the following are true:

- The preliminary LIBERO result is repeated with uncertainty, and the proposed method beats the current IDP port, prefetch-only, blending, and matched endpoint/proximal baselines on aggregate success or continuity.
- Prefix conditioning improves boundary continuity and does not reduce zero-prefix LIBERO performance.
- The result holds across LIBERO suites and the physical soft-bag task.
- GR00T flow, pi0.5 RTC, and matched one-step baselines are trained and evaluated under documented controlled and best-system protocols.
- The latency study contains a same-stack GR00T comparison and live-loop metrics.
- Soft-bag gains appear in the equal-server native-speed comparison and are explained by shared-rate and delay-matched controls, command-age distributions, continuity, and task-phase metrics.
- Drifting demonstrates lower same-server peak/steady-state GPU memory than the claimed baselines, with same-backbone head savings separated from whole-model-family differences.
- Every table can be regenerated from saved raw results and a frozen configuration.
- The paper explicitly distinguishes inherited IDP supervision from the new overlap-conditioned action head and training procedure.
- Negative results and failure modes, including mode collapse and VLM-dominated latency, are reported.

## Expected repository surfaces during execution

- `src/lerobot/policies/drifting/`: retained unconditioned IDP-port baseline for ablation.
- `src/lerobot/policies/drif_ov/`: prefix-aware inputs, overlap training, one-step successor generation, diagnostics, and checkpointed ablation configuration.
- `tests/policies/drifting/`: inherited IDP equation and unconditioned-baseline tests.
- `tests/policies/drif_ov/`: registration, prefix masking, exact preservation, one-evaluation inference, boundary objectives, runtime prefix construction, and baseline-equivalence tests.
- `tests/policies/drifting/benchmark_inference_latency.py`: configurable and raw-result-producing benchmark.
- `docs/source/drifting.mdx`: inherited baseline documentation.
- `docs/source/drif_ov.mdx`: precise inherited-versus-new method description and reproducible usage.
- Experiment configs/scripts and result-processing code using existing repository conventions.
- `0724inference_analysis.md`: retained as preliminary evidence, then superseded by a publication-grade comparative report.

## Notes

- The source paper itself is inconsistent about self-neighbors: its conditional covariance equation excludes `j=i`, while its implementation table includes `j=i`. Treat this as an empirical choice, not a fidelity error.
- Prefix conditioning is only publishable if it beats simpler prefetch and blending controls; architectural difference alone is not enough.
- If the full IDP geometry objective adds no benefit after prefix conditioning, report that result and frame IDP as the initialization point rather than forcing geometry into the final method.
