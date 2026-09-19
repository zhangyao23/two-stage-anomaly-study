# Failure analysis and claim boundaries

## Detection is not diagnosis

Reconstruction learning minimizes normal-data distortion, not anomaly-type classification loss.
The synthetic results establish only that the particular frozen 8-dimensional representation plus
small MLP performs worse than the raw-input MLP under this configuration. There is no
capacity-matched probe, latent rescaling control, bottleneck sweep, balanced loss comparison or
information-theoretic test. Claims of irreversible information loss, universal denoising benefits
or representation superiority would exceed the evidence.

The generator creates equal event counts, not equal window counts. A three-step spike produces
far fewer anomalous windows than a 48-step drift. A short window can also observe only the edge
of a shift/drift. These two effects, combined with unweighted cross-entropy and limited training,
are plausible contributors to poor classification. They have not been isolated causally.

## Exact propagation, not a product of unrelated metrics

For type k, `P(final=k | truth=k) = P(gate=1 | truth=k) × P(head=k | gate=1, truth=k)`.
The head's independent accuracy over every true k example is not the second term. The detector
may selectively pass easier or harder examples. All test predictions are saved, allowing either
conditional or unconditional quantities to be recomputed.

At seed 17, raw-head drift recall is 0.663 on all true synthetic drift windows but falls to 0.374 in the
complete AE cascade. In `test-0:136:152`, all three heads independently say drift, yet the gate
rejects the window. A downstream classifier cannot rescue an input it never sees.

Normal false alarms have the opposite path. The three-class head has no normal/reject output,
so `test-0:8:24` is inevitably assigned an abnormal label after the detector flags it. This is a
structural limitation of the evaluated design, not an implementation bug hidden from scoring.

## Public data failure and calibration

Public AE test FPR is 0.292, compared with a nominal calibration tail of roughly 0.05. This
contradicts a future-FPR guarantee but does not contradict empirical quantile calibration itself.
The per-series chronological protocol exposes changes in observed normal score distributions.
Broad benchmark anomaly windows, imperfect normal labeling, short histories and univariate
model limitations may all contribute. No ablation isolates their relative effects.

The statistical baseline achieves pooled F1 0.242 versus AE 0.239 and AP 0.172 versus 0.152.
Neither is competitive evidence for deployment; no significance claim is made from this single
public initialization. The comparison is kept even though it does not favor the neural model.

Twelve of the 17 test segments contain no positive labels. Their F1 convention is zero and AP
undefined; they still contribute normal windows and false alarms. Reporting only files with
anomalies would silently change the test distribution. Public prediction CSVs record every
held-out window's source span, binary truth, score and threshold without redistributing measurements.

## Example selection and verification

Synthetic examples are the first matching test window for each error category in each seed.
Both directions of raw-versus-encoded disagreement are retained. `failures_17.json` contains
the actual generated window values; `failures.png` plots them. `results/public/results.json`
contains first FN/FP source spans per detector and series. These are diagnostic examples, not
an unbiased sample of all error mechanisms; full CSV predictions remain the basis of metrics.

Scores, threshold comparisons, total counts and confusion matrices are audited by
`scripts/audit_results.py`. Test labels do not alter model training or calibration, verified by a
test that changes all test labels and checks model weights and thresholds for exact equality.

## Controls deliberately left for future work

- Match head parameter counts as well as epoch/sample budgets, and normalize frozen embeddings.
- Compare class weighting or balanced sampling selected using validation, without reusing the
  current test results for model selection.
- Vary temporal context, anomaly strength and generator families with a newly declared protocol.
- Add second-stage rejection/normal labels and evaluate its trade-off rather than assuming it helps.
- Study adaptive calibration under drift with a prospective evaluation protocol.

None is presented as implemented or as evidence of a novel method. Streaming/threading was
optional and is not included: its omission does not affect the core offline research question.
