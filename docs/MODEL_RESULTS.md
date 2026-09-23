# Baseline model results

## Evaluation contract

- Dataset: public WM-811K wafer maps
- Total maps profiled: 811,457
- Labelled classes retained: 9
- Training samples: 54,355
- Test samples: 118,595
- Split: WM-811K official training/test assignment
- Model: 300-tree `RandomForestClassifier`
- Inputs: nine deterministic engineered wafer-map features
- Class handling: `balanced_subsample`

The 638,507 unlabelled maps are available for inspection but are excluded from supervised model
training and evaluation.

## Aggregate results

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8594 |
| Macro-F1 | 0.4128 |
| Balanced accuracy | 0.4391 |
| Weighted F1 | 0.8829 |

Accuracy and weighted F1 are dominated by the `none` class, which represents 110,701 of 118,595
test samples. Macro-F1 and balanced accuracy are therefore the primary baseline metrics.

## Per-class results

| Class | Precision | Recall | F1 | Test support |
| --- | ---: | ---: | ---: | ---: |
| Center | 0.0699 | 0.1887 | 0.1020 | 832 |
| Donut | 0.1111 | 0.0274 | 0.0440 | 146 |
| Edge-Loc | 0.1455 | 0.4221 | 0.2164 | 2,772 |
| Edge-Ring | 0.5320 | 0.5613 | 0.5462 | 1,126 |
| Loc | 0.1564 | 0.2230 | 0.1839 | 1,973 |
| Near-full | 0.9592 | 0.9895 | 0.9741 | 95 |
| Random | 0.8050 | 0.6265 | 0.7046 | 257 |
| Scratch | 0.0143 | 0.0173 | 0.0157 | 693 |
| none | 0.9623 | 0.8966 | 0.9283 | 110,701 |

## Error analysis

The baseline works best when the defect pattern is strongly represented by global statistics:
`Near-full`, `Random`, `none` and, to a lesser extent, `Edge-Ring`. It performs poorly on patterns
whose identity depends on richer spatial geometry. In particular:

- 590 of 832 `Center` wafers were predicted as `none`.
- 508 of 693 `Scratch` wafers were predicted as `none`.
- Only 4 of 146 `Donut` wafers were correctly classified.
- 1,364 of 2,772 `Edge-Loc` wafers were predicted as `none`.
- Class balancing raises minority-class recall but also creates false positives from the dominant
  `none` class, including 6,276 `none` wafers predicted as `Edge-Loc`.

These errors support a concrete next hypothesis: global aggregate features are insufficient for
orientation, rings, scratches and local clusters. A spatial model should be compared against this
baseline using the same official split, without changing the evaluation contract.

## Next experiment

1. Normalize variable-size wafer maps to a fixed spatial representation.
2. Train a class-weighted CNN or an alternative spatial-feature model.
3. Compare macro-F1, balanced accuracy, per-class recall and latency against this baseline.
4. Inspect regressions by class rather than selecting a model on accuracy alone.
5. Keep the Random Forest as an interpretable fallback and serving-contract reference.

## Claim boundary

These results measure labelled wafer-pattern classification. WM-811K does not contain equipment
telemetry, chamber history, recipes, maintenance events or confirmed physical root causes. The
investigation assistant therefore provides review hypotheses and similar cases, not causal proof.


## Spatial CNN experiment

### Hypothesis

The Random Forest uses nine global engineered features. Those features summarize geometry and failure density but discard the exact spatial arrangement of failing dies. A compact CNN can consume the wafer-map topology directly and may therefore recover patterns such as rings, scratches and localized clusters.

### Experimental contract

- Same labelled WM-811K records
- Same official train/test assignment
- Fixed 32x32 nearest-neighbour spatial representation
- Three convolutional blocks with global average pooling
- Class-weighted cross-entropy
- AdamW optimizer
- Seeded training
- Accuracy, macro-F1, balanced accuracy and per-class results
- Random Forest remains the reference baseline

### Results

CNN results are intentionally **not hard-coded before the experiment is run**. Run:

```bash
python -m scripts.train_cnn
```

Then inspect:

```text
artifacts/cnn/metrics.json
```

The next review should compare macro-F1, balanced accuracy and minority-class recall, then inspect any regressions by class. No model should be selected from accuracy alone.

### Claim boundary

WM-811K supports wafer-pattern classification only. It does not provide equipment telemetry, recipes, chamber histories, maintenance events or confirmed physical root causes.
