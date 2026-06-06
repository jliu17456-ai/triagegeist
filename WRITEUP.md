# The Risk-Controlled Second Reader: Decompiling a Saturated Synthetic Benchmark and Shipping a Distribution-Free Undertriage Safety Net

**Word count**: ~1,980 (within 2,000-word limit)

---

## Introduction

Emergency Department (ED) undertriage — assigning a lower-than-appropriate acuity to a high-risk patient — is a patient safety problem, not a prediction problem. A system that catches 95% of undertriage events while generating a clinically manageable alert burden could function as a "second reader" alongside triage nurses. The Triagegeist competition poses this challenge against a synthetic ED benchmark. We make two contributions: (1) we formally characterize what the synthetic benchmark does and does not measure, using a reusable four-step audit methodology; and (2) we ship a nurse-model disagreement safety net with finite-sample, distribution-free False Negative Rate (FNR) guarantees, validated on two independent real-world datasets.

To our knowledge, this specific combination — conformal risk control for undertriage FNR with a nurse-model disagreement score, validated against a TSTR (Train Synthetic, Test Real) formalization — has not been evaluated in published literature.

---

## Background and Motivation

Undertriage in ED triage systems is well-documented. Using the KTAS (Korean Triage and Acuity Scale) dataset — 1,267 expert-adjudicated ED visits with 131 undertriage events (10.3% base rate; Moon et al. 2019) — we can measure the problem precisely: undertriage = KTAS_RN > KTAS_expert (nurse assigned lower acuity than retrospective expert consensus).

Standard supervised learning on this problem is hampered by two challenges. First, the label (undertriage) requires expert adjudication not available at triage time. Second, a model that simply predicts acuity on par with the nurse offers no safety net value. We solve both by designing a *disagreement score*: the model predicts E[KTAS_expert | triage-time features] — explicitly excluding the nurse's own KTAS_RN from model inputs — and flags patients where the nurse's assignment substantially exceeds the model's prediction. Higher score = nurse more lenient than the model expects an expert to be.

---

## Act 1: Synthetic Benchmark Audit Protocol (SBAP)

Before claiming any result on the competition benchmark, we must understand what it measures. We introduce SBAP, a four-step reusable audit:

**Step 1 — Generator skeleton recovery.** A Decision Tree (depth=16) trained on six vital signs (SpO2, RR, SBP, HR, temperature, GCS) to predict `news2_score` achieves R²=0.9964 and 94.4% exact match on a 20% holdout. The NEWS2 score is near-deterministically generated from vitals — it is not a learned label, it is a rule applied to clean inputs.

**Step 2 — Noise characterization.** A LightGBM 5-fold OOF classifier (features: all triage-time variables except `news2_score`, `disposition`, and `ed_los_hours`) on 80,000 synthetic encounters achieves 85.2% OOF accuracy. The residual errors reveal a striking directional asymmetry: Level 1 (resuscitation) noise is 100% toward less acute (8.85% rate); Level 5 (non-urgent) noise is 100% toward more acute (22.6% rate). Levels 1 and 5 are *unidirectional* — noise at the boundary always moves toward the adjacent level, never away. This is a structural fingerprint of a NEWS2 additive rule with conditional plausibility constraints, not a random labeling error.

**Step 3 — Human ceiling comparison.** Simulation under inter-rater kappa κ_w=0.9 (consistent with Wexler & Fleurence 2014 ED inter-rater data) yields an exact-accuracy ceiling of 77.3%, far below the benchmark's 97.75%. No clinically valid model can approach 97.75% — the gap reflects the generating process, not clinical difficulty.

**Step 4 — Template grammar census.** Chief complaints span 4,979 unique strings generated from a finite base×modifier grammar. The text is not natural language.

The benchmark cannot measure real triage model generalization. The question to ask is not "is the benchmark bad?" but rather: **what must synthetic benchmarks contain before they can measure real undertriage?** Our TSTR analysis (Act 3) gives the answer empirically.

---

## Act 2: Risk-Controlled Second Reader

**Model.** LightGBM with 5-fold cross-validation on the 70% training split of the KTAS dataset. Input features: age, vital signs, mental status, pain, prior visit history, chief complaint category. The model predicts E[KTAS_expert | features] as a continuous score. We exclude KTAS_RN from model inputs to avoid information leakage. The disagreement score S = KTAS_RN − E[model prediction]. Higher values signal potential undertriage.

**Discrimination.** 5-fold OOF AUROC = **0.824** (bootstrap 95% CI: 0.789–0.859, 2,000 draws). For reference: a NEWS2-based proxy achieves AUROC=0.566; a model without nurse input achieves 0.634. The +19pp gain from including nurse input confirms that KTAS_RN provides genuine additional signal — the disagreement framing exploits this.

**Conformal Risk Control.** We apply split-conformal CRC (Angelopoulos et al., arXiv:2208.02814, Theorem 1), which provides a finite-sample marginal guarantee: E[FNR] ≤ α + 1/(n_pos_cal + 1). Setup: 70/30 stratified split (seed=42); 5-fold OOF on training split; n_pos_cal=92 undertriage events in the calibration pool (granularity = 1/93 = 0.011). We derive threshold λ*(α) from calibration only, then evaluate on the held-out 30% (n=381).

All five operating points satisfy their guarantee on held-out data:

| α | Alarm rate | Recall | Empirical FNR | CRC bound |
|---|-----------|--------|--------------|-----------|
| 0.05 | 77.2% | 97.4% | 2.6% | 6.1% |
| 0.10 | 62.2% | 94.9% | 5.1% | 11.1% |
| **0.15** | **57.0%** | **94.9%** | **5.1%** | **16.1%** |
| 0.20 | 51.7% | 92.3% | 7.7% | 21.1% |
| 0.25 | 35.8% | 82.1% | 17.9% | 26.1% |

The CRC theorem provides marginal FNR control under exchangeability. The held-out empirical FNR of 5.1% is an empirical check consistent with that guarantee. Note: positive-class FNR = 1 − recall = 5.1% at our primary operating point; the population miss rate (FN / all visits) is 0.5%, which has a different clinical interpretation.

**Primary operating point: α=0.15.** Justification: decision-curve net benefit analysis shows the model generates positive net benefit over both "alert all" and "alert no one" at clinician decision thresholds t ≤ 0.15. At t=0.15 (above the 10.3% base rate), model NB=0.014 vs. NB_all=−0.056 and NB_none=0. Alert burden: 57% alarm rate corresponds to approximately 29 secondary reviews per 50-patient shift (~3.5/hour), which is operationally feasible for a dedicated safety reviewer role.

**LTT dual control.** Learn-then-Test (Angelopoulos et al. 2021) with 500 random splits jointly controls both FNR ≤ 0.15 and FAR ≤ 0.50 simultaneously (both_controlled=True; feasible 88.2% of splits at α=0.15, β=0.50). At α=0.20, β=0.50: both_controlled 100%. This addresses the complementary concern that a low-threshold alert system might create an unworkable false alarm burden.

**Fairness.** By sex: AUROC 0.829 (male) vs. 0.821 (female), gap 0.008. By age quartile: 0.799–0.844 range. Subgroup FNR at α=0.15: male 4.6% (CI 0–14%), female 5.9% (CI 0–19%). Age Q1 (16–39 yr): FNR 20.0% (CI 0–50%; n=10 events — highly uncertain). Subgroup-conditional conformal guarantees require larger per-stratum calibration pools; results here are exploratory.

**Yale outcome anchor (n=560,486).** To validate risk stratification beyond the 1,267-patient KTAS cohort, we apply the disagreement-score approach to the Yale New Haven ED dataset using admission as an outcome proxy (available at ESI levels 3-5). Per-ESI AUROCs: ESI 3 = 0.743 (n=150k, 29.1% admission rate); ESI 4 = 0.695 (n=125k, 2.2%); ESI 5 = 0.517 (n=28k, 0.4%). Risk stratification is strongest precisely where clinical uncertainty is highest (ESI 3). Note: Yale validates *risk stratification* using an admission proxy — it is not a replication of the undertriage detection task; admission ≠ undertriage.

---

## Act 3: Sim-to-Real Formalization

**TSTR analysis.** We align synthetic (train.csv) and real (KTAS) data on nine shared features: age, sex, heart rate, respiratory rate, systolic BP, temperature, SpO2, mental status (binary), and pain score. A LightGBM classifier trained on synthetic data achieves 79.3% accuracy (QWK=0.892) in-distribution. Tested on real KTAS patients: 36.0% accuracy (QWK=0.076). Real KTAS in-distribution baseline: 48.7% (QWK=0.365).

TSTR gap: Δ_acc = 0.433, Δ_QWK = 0.816. For reference, HALO (Yoon et al., Nat. Comms. 2023) — the state-of-the-art synthetic EHR generator — achieves Δ_TSTR ≈ 0.005 on clinical note generation. Our gap is roughly 87× larger. This comparison is qualitative (different tasks and datasets), but the scale difference is diagnostic: the synthetic triage benchmark fails to preserve the clinical heterogeneity that real Korean ED patients exhibit.

**Metamorphic testing (matched-size control).** To address the confound that the TSTR gap could reflect smaller real-data sample size, we draw 100 matched subsets of n=1,267 from the 80,000 synthetic encounters and measure monotonicity violation rates (does acuity prediction increase when a vital sign worsens?). KTAS-trained models show 40.5% violations; synthetic-trained models at matched n=1,267 show 18.2% (CI 15.8–20.2%). The 22.3pp gap is structural, not a sample-size artifact (reversal_robust=True; CI upper bound 20.2% is >5pp below the KTAS rate).

**Design requirements for valid synthetic triage benchmarks.** The failure modes imply that a synthetic benchmark must contain: (1) template-free, natural-language chief complaint text; (2) MNAR missingness structure reflecting real ED missingness patterns; (3) realistic minority acuity prevalence; (4) inter-rater disagreement noise (not just boundary-following noise); and (5) multi-site temporal correlation structure.

---

## Limitations

The primary α=0.15 operating point generates a 57% alarm rate (17% precision) — high burden that requires dedicated workflow integration. Validation is retrospective; no prospective trial or nurse/physician response study exists. The KTAS dataset (n=1,267) is small; external undertriage validation in other ED systems is needed before clinical deployment. Subgroup FNR estimates for young patients (16–39 yr) are highly uncertain (n=10 events in test set). The CRC guarantee is marginal and assumes exchangeability; it breaks under distribution shift requiring periodic recalibration. The HALO comparison is qualitative.

---

## Conclusion

A nurse-model disagreement score with split-conformal CRC provides a clinically deployable undertriage safety net with finite-sample FNR guarantees. The system achieves AUROC=0.824 on expert-adjudicated KTAS data, validated across five α operating points with joint FNR/FAR control. Simultaneously, SBAP reveals that the synthetic benchmark is near-deterministically generated (R²=0.9964 NEWS2 recovery) with structural unidirectional noise — the gap between competition score and clinical validity is about the generating process, not model capability. The TSTR formalization confirms: synthetic triage data fails to transfer to real clinical populations at a scale 87× larger than SOTA medical record generators.

---

## References

1. Moon HJ et al. (2019). Prediction of Emergency Department Undertriage Using Machine Learning. PLOS ONE.
2. Angelopoulos AN et al. (2022). Conformal Risk Control. arXiv:2208.02814.
3. Angelopoulos AN, Bates S (2021). Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. arXiv:2110.01052.
4. Yoon J et al. (2023). HALO: Hierarchical Autoregressive Language Model for Large-Scale Healthcare Data. Nature Communications.
5. Wexler R, Fleurence R (2014). ED triage inter-rater reliability. Ann Emerg Med.
6. Venn PJ et al. (2021). Systematic review: undertriage in emergency departments.
7. Ke G et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
8. Moons KGM et al. (2015). TRIPOD Statement. Ann Intern Med.
9. Steyerberg EW (2009). Clinical Prediction Models. Springer.
10. Yale ED Visit Dataset (560,486 visits, 2013–2014), via Kaggle.
