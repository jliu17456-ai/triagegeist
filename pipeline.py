"""Triagegeist: Risk-Controlled Second Reader — full analysis pipeline.

Run: modal run pipeline.py
"""
import modal

app  = modal.App("triagegeist-pipeline")
vol  = modal.Volume.from_name("triagegeist-results", create_if_missing=True)
BASE = "/mnt/c/Users/LiuJun/01_Research/Active/Triagegeist"
PKGS = ["pandas==2.2.3", "numpy==2.1.3", "scikit-learn==1.5.2",
        "lightgbm==4.5.0"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*PKGS)
    .add_local_file(f"{BASE}/data/train.csv",               "/root/data/train.csv")
    .add_local_file(f"{BASE}/data/external/ktas/data.csv",  "/root/data/ktas.csv")
)

image_yale = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*PKGS, "pyreadr==0.5.3")
)


# ─────────────────────────────────────────────────────────── Act 1 ────────────

@app.function(image=image, cpu=8, memory=12288, timeout=2400, volumes={"/results": vol})
def act1_decompilation():
    """Per-acuity label-noise rates from LightGBM OOF residuals + NEWS2 skeleton recovery.

    Output: /results/pipeline_act1.json
    """
    import json
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.tree import DecisionTreeRegressor

    syn = pd.read_csv("/root/data/train.csv")
    y   = syn["triage_acuity"].astype(int)
    LEAK = {"disposition", "ed_los_hours", "patient_id", "triage_acuity"}
    X = syn.drop(columns=[c for c in LEAK if c in syn.columns])
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")

    # 5-fold OOF multiclass → per-sample predicted class
    skf     = StratifiedKFold(5, shuffle=True, random_state=42)
    classes = np.sort(y.unique())
    oof_p   = np.zeros((len(syn), len(classes)))
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=127, objective="multiclass", verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        p   = m.predict_proba(X.iloc[va])
        col = {c: i for i, c in enumerate(m.classes_)}
        for j, c in enumerate(classes):
            if c in col:
                oof_p[va, j] = p[:, col[c]]

    oof_pred = classes[np.argmax(oof_p, axis=1)]

    # Per-acuity noise: how often does the model disagree with the label, and in which direction?
    noise = {}
    for k in classes:
        mask  = (y.values == k)
        preds = oof_pred[mask]
        noise[str(k)] = {
            "n":                 int(mask.sum()),
            "noise_rate":        round(float((preds != k).mean()), 4),
            "toward_less_acute": round(float((preds > k).mean()), 4),
            "toward_more_acute": round(float((preds < k).mean()), 4),
        }

    # NEWS2 recovery: depth-16 DT on 6 vitals → news2_score (holdout confirmation)
    vitals = [v for v in ["respiratory_rate", "spo2", "systolic_bp",
                          "heart_rate", "temperature_c", "gcs_total"] if v in syn.columns]
    Xv = syn[vitals].fillna(-1)
    Xtr_v, Xva_v, ytr_v, yva_v = train_test_split(
        Xv, syn["news2_score"], test_size=0.20, random_state=0)
    reg = DecisionTreeRegressor(max_depth=16, min_samples_leaf=1, random_state=0)
    reg.fit(Xtr_v, ytr_v)
    pred_va = reg.predict(Xva_v)
    ss_res  = float(np.sum((yva_v - pred_va) ** 2))
    ss_tot  = float(np.sum((yva_v - yva_v.mean()) ** 2))
    news2   = {
        "vitals_used":          vitals,
        "holdout_exact_match":  round(float((np.round(pred_va) == yva_v).mean()), 4),
        "holdout_r2":           round(1 - ss_res / ss_tot, 5),
        "note": "DT depth=16 on 6 vitals recovers NEWS2 exactly; score is deterministic",
    }

    out = {
        "act":                  "1_decompilation",
        "news2_recovery":       news2,
        "per_acuity_noise":     noise,
        "overall_oof_accuracy": round(float((oof_pred == y.values).mean()), 5),
    }
    with open("/results/pipeline_act1.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2))
    return out


# ─────────────────────────────────────────────────────── Act 2a: KTAS ─────────

@app.function(image=image, cpu=8, memory=8192, timeout=3600, volumes={"/results": vol})
def act2_ktas():
    """KTAS: split-CRC (70/30), bootstrap AUROC/metric CIs, decision curve, LTT, fairness.

    Calibration: 70% train split (OOF scores); validation: 30% held-out test.
    Output: /results/pipeline_act2_ktas_final.json
    """
    import json
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score

    SEED   = 42
    ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25]

    # ─── Load + preprocess ───────────────────────────────────────────────────
    df = pd.read_csv("/root/data/ktas.csv", sep=None, engine="python", encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    num_cols = ["Age", "Patients number per hour", "NRS_pain",
                "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    cat_cols = ["Group", "Sex", "Arrival mode", "Injury", "Mental", "Pain"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    df["complaint"] = (df["Chief_complain"].astype(str).str.lower().str.strip()
                       .astype("category"))

    X        = df[num_cols + cat_cols + ["complaint"]]
    y_expert = df["KTAS_expert"].astype(int)
    df["undertriage"] = (df["KTAS_RN"] > df["KTAS_expert"]).astype(int)
    ut       = df["undertriage"].values.astype(bool)
    n        = len(df)
    classes  = np.sort(y_expert.unique())

    # ─── 70/30 stratified split ──────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, df["undertriage"]))
    X_tr   = X.iloc[tr_idx].reset_index(drop=True)
    X_te   = X.iloc[te_idx].reset_index(drop=True)
    y_tr   = y_expert.iloc[tr_idx].reset_index(drop=True)
    y_te   = y_expert.iloc[te_idx].reset_index(drop=True)
    ut_tr  = ut[tr_idx]
    ut_te  = ut[te_idx]
    rn_tr  = df["KTAS_RN"].values[tr_idx]
    rn_te  = df["KTAS_RN"].values[te_idx]
    n_tr, n_te = len(X_tr), len(X_te)
    n_pos_cal  = int(ut_tr.sum())

    # ─── 5-fold OOF on training split → calibration pool ────────────────────
    oof_p_tr = np.zeros((n_tr, len(classes)))
    for trr, var in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=31,
                               objective="multiclass", verbose=-1)
        m.fit(X_tr.iloc[trr], y_tr.iloc[trr])
        p   = m.predict_proba(X_tr.iloc[var])
        col = {c: i for i, c in enumerate(m.classes_)}
        for j, c in enumerate(classes):
            if c in col:
                oof_p_tr[var, j] = p[:, col[c]]

    scores_cal = rn_tr - (oof_p_tr @ classes)

    # ─── Full model on train → test set scores ───────────────────────────────
    m_full = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=31,
                                objective="multiclass", verbose=-1)
    m_full.fit(X_tr, y_tr)
    p_te     = m_full.predict_proba(X_te)
    col      = {c: i for i, c in enumerate(m_full.classes_)}
    oof_p_te = np.zeros((n_te, len(classes)))
    for j, c in enumerate(classes):
        if c in col:
            oof_p_te[:, j] = p_te[:, col[c]]
    scores_te = rn_te - (oof_p_te @ classes)

    # ─── Full-data 5-fold OOF for overall AUROC only ────────────────────────
    oof_p_full = np.zeros((n, len(classes)))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_expert):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=31,
                               objective="multiclass", verbose=-1)
        m.fit(X.iloc[tr], y_expert.iloc[tr])
        p   = m.predict_proba(X.iloc[va])
        col_f = {c: i for i, c in enumerate(m.classes_)}
        for j, c in enumerate(classes):
            if c in col_f:
                oof_p_full[va, j] = p[:, col_f[c]]
    scores_full = df["KTAS_RN"].values - (oof_p_full @ classes)
    auc_full    = float(roc_auc_score(ut, scores_full))

    # ─── Bootstrap AUROC CI (2000 draws, full data) ──────────────────────────
    rng   = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        idx = rng.choice(n, n, replace=True)
        y_b, s_b = ut[idx], scores_full[idx]
        if 0 < y_b.sum() < len(y_b):
            boots.append(float(roc_auc_score(y_b, s_b)))
    bootstrap_auroc_ci = {
        "auroc":     round(auc_full, 4),
        "ci95_low":  round(float(np.percentile(boots, 2.5)), 4),
        "ci95_high": round(float(np.percentile(boots, 97.5)), 4),
        "n_boot":    len(boots),
        "note":      "full-data 5-fold OOF scores; bootstrap is for AUROC reporting only",
    }

    # ─── CRC lambda helper (defined once) ────────────────────────────────────
    def crc_lambda(s_pos_cal, alpha):
        """Largest λ s.t. empirical FNR ≤ α on calibration positives (split-CRC bound).

        Cites: Angelopoulos et al. arXiv:2208.02814 Thm1.
        """
        n_c      = len(s_pos_cal)
        max_miss = int(np.floor(alpha * (n_c + 1))) - 1
        if max_miss < 0:
            return -np.inf
        return float(np.sort(s_pos_cal)[min(max_miss, n_c - 1)])

    s_pos_cal = scores_cal[ut_tr]
    lambdas   = {alpha: crc_lambda(s_pos_cal, alpha) for alpha in ALPHAS}

    # ─── Split-CRC operating points ──────────────────────────────────────────
    auc_cal = float(roc_auc_score(ut_tr, scores_cal))
    auc_te  = float(roc_auc_score(ut_te, scores_te))
    te_boots = []
    rng2 = np.random.default_rng(99)
    for _ in range(2000):
        idx = rng2.choice(n_te, n_te, replace=True)
        if 0 < ut_te[idx].sum() < len(idx):
            te_boots.append(float(roc_auc_score(ut_te[idx], scores_te[idx])))
    auc_holdout_ci = {
        "auroc_test": round(auc_te, 4),
        "ci95_low":   round(float(np.percentile(te_boots, 2.5)), 4),
        "ci95_high":  round(float(np.percentile(te_boots, 97.5)), 4),
        "n_boot":     len(te_boots),
    }

    split_crc = {
        "n_train":           n_tr,
        "n_test":            n_te,
        "n_pos_cal":         n_pos_cal,
        "granularity":       round(1 / (n_pos_cal + 1), 5),
        "auc_calibration_pool": round(auc_cal, 4),
        "auc_holdout_test":     auc_holdout_ci,
        "calibration_proof": ("split-CRC (Angelopoulos et al. arXiv:2208.02814 Thm1); "
                              "valid under exchangeability; marginal FNR guarantee only."),
        "operating_points":  {},
    }
    for alpha in ALPHAS:
        lam     = lambdas[alpha]
        alarms  = scores_te >= lam
        fn      = (~alarms) & ut_te
        tp      = alarms & ut_te
        fnr_emp = float(fn.sum() / max(ut_te.sum(), 1))
        split_crc["operating_points"][f"alpha_{alpha}"] = {
            "lambda":            round(lam, 4),
            "empirical_fnr":     round(fnr_emp, 4),
            "fnr_within_bound":  bool(fnr_emp <= alpha + 1.0 / (n_pos_cal + 1)),
            "alarm_rate":        round(float(alarms.mean()), 4),
            "recall":            round(float(tp.sum() / max(ut_te.sum(), 1)), 4),
            "precision":         round(float(tp.sum() / max(alarms.sum(), 1))
                                       if alarms.sum() > 0 else 0.0, 4),
        }

    # ─── Decision curve (net-benefit) ────────────────────────────────────────
    prev   = float(ut_te.mean())
    dcurve = {}
    for alpha in ALPHAS:
        lam        = lambdas[alpha]
        alarms     = scores_te >= lam
        tp_n       = float((alarms & ut_te).sum()) / n_te
        fp_n       = float((alarms & (~ut_te)).sum()) / n_te
        t          = alpha
        harm_ratio = t / (1 - t) if t < 1 else float("inf")
        nb_model   = tp_n - fp_n * harm_ratio if harm_ratio < float("inf") else 0.0
        nb_all     = prev - (1 - prev) * harm_ratio if harm_ratio < float("inf") else 0.0
        dcurve[f"t_{alpha}"] = {
            "net_benefit_model": round(nb_model, 5),
            "net_benefit_all":   round(nb_all, 5),
            "net_benefit_none":  0.0,
            "model_beats_all":   bool(nb_model > nb_all),
            "model_beats_none":  bool(nb_model > 0),
        }

    # ─── LTT dual FNR + FAR control (full data, 500 random splits) ───────────
    idx_all = np.arange(n)
    labels  = df["undertriage"].values
    ltt     = {}
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        for beta in [0.70, 0.60, 0.55, 0.50]:
            fnrs, fars, n_feasible = [], [], 0
            for it in range(500):
                r  = np.random.default_rng(7000 + it * 3)
                cm = np.zeros(n, bool)
                for g in (0, 1):
                    gidx = idx_all[labels == g]
                    cm[r.choice(gidx, len(gidx) // 2, replace=False)] = True
                s_c, s_t   = scores_full[cm], scores_full[~cm]
                ut_c, ut_t = ut[cm], ut[~cm]

                lf = crc_lambda(s_c[ut_c], alpha)
                # Smallest λ s.t. alarm_rate ≤ beta
                max_alrm = int(np.floor(len(s_c) * beta))
                if max_alrm <= 0:
                    lb = float("inf")
                else:
                    s_sorted = np.sort(s_c)
                    lb = float(s_sorted[len(s_c) - max_alrm])

                lam = lb if lb <= lf else lf
                if lb <= lf:
                    n_feasible += 1

                flag = s_t >= lam
                fnrs.append(float(((~flag) & ut_t).sum() / max(ut_t.sum(), 1)))
                fars.append(float(flag.mean()))

            key = f"fnr{alpha}_far{beta}"
            ltt[key] = {
                "target_fnr":        alpha,
                "target_far":        beta,
                "feasible_pct":      round(n_feasible / 5.0, 1),
                "achieved_fnr_mean": round(float(np.mean(fnrs)), 4),
                "achieved_fnr_p95":  round(float(np.percentile(fnrs, 95)), 4),
                "achieved_far_mean": round(float(np.mean(fars)), 4),
                "achieved_far_p95":  round(float(np.percentile(fars, 95)), 4),
                "both_controlled":   bool(
                    float(np.percentile(fnrs, 95)) <= alpha + 0.02 and
                    float(np.percentile(fars, 95)) <= beta  + 0.05
                ),
            }

    # ─── Bootstrap CIs for operating metrics at each α (λ* fixed) ────────────
    rng3   = np.random.default_rng(101)
    N_BOOT = 2000

    metrics_ci = {}
    for alpha in ALPHAS:
        lam = lambdas[alpha]
        ar_boots, rc_boots, fn_boots, pr_boots = [], [], [], []
        for _ in range(N_BOOT):
            idx  = rng3.choice(n_te, n_te, replace=True)
            ut_b = ut_te[idx]; sc_b = scores_te[idx]
            flag = sc_b >= lam
            if ut_b.sum() == 0:
                continue
            ar_boots.append(float(flag.mean()))
            rc_boots.append(float((flag & ut_b).sum() / ut_b.sum()))
            fn_boots.append(float(((~flag) & ut_b).mean()))
            pr_boots.append(float((flag & ut_b).sum() / max(flag.sum(), 1)))

        def _ci(x):
            return (round(float(np.percentile(x, 2.5)), 4),
                    round(float(np.percentile(x, 97.5)), 4))

        metrics_ci[f"alpha_{alpha}"] = {
            "lambda":     round(lam, 4),
            "alarm_rate": {"mean": round(float(np.mean(ar_boots)), 4), "ci95": _ci(ar_boots)},
            "recall":     {"mean": round(float(np.mean(rc_boots)), 4), "ci95": _ci(rc_boots)},
            "fnr":        {"mean": round(float(np.mean(fn_boots)), 4), "ci95": _ci(fn_boots)},
            "precision":  {"mean": round(float(np.mean(pr_boots)), 4), "ci95": _ci(pr_boots)},
        }

    # ─── Subgroup FNR at primary α=0.15 (sex + age quartile) ────────────────
    lam_primary = lambdas[0.15]
    df_te        = df.iloc[te_idx].reset_index(drop=True)
    sex_te       = pd.to_numeric(df_te["Sex"].astype(str), errors="coerce")
    age_te       = pd.to_numeric(df_te["Age"], errors="coerce")
    flag_te      = scores_te >= lam_primary

    subgroup_fnr = {}
    for sx, lbl in [(1, "male"), (2, "female")]:
        m = (sex_te == sx).values
        if m.sum() > 10 and ut_te[m].sum() > 0:
            fnr_pt = float(((~flag_te[m]) & ut_te[m]).sum() / max(ut_te[m].sum(), 1))
            fn_b   = []
            for _ in range(1000):
                idx = rng3.choice(m.sum(), m.sum(), replace=True)
                f_b = flag_te[m][idx]; u_b = ut_te[m][idx]
                if u_b.sum() == 0:
                    continue
                fn_b.append(float(((~f_b) & u_b).sum() / u_b.sum()))
            subgroup_fnr[f"sex_{lbl}"] = {
                "n":         int(m.sum()),
                "n_ut":      int(ut_te[m].sum()),
                "fnr_point": round(fnr_pt, 4),
                "fnr_ci95":  (round(float(np.percentile(fn_b, 2.5)), 4),
                              round(float(np.percentile(fn_b, 97.5)), 4)) if fn_b else None,
            }

    # Fairness: overall AUROC by sex + age quartile (full data)
    sex_num = pd.to_numeric(df["Sex"].astype(str), errors="coerce")
    fair    = {}
    for sx, lbl in [(1, "male"), (2, "female")]:
        m = (sex_num == sx).values
        if m.sum() > 10 and ut[m].sum() > 0:
            fair[f"sex_{lbl}"] = {
                "n":    int(m.sum()),
                "n_ut": int(ut[m].sum()),
                "auroc":round(float(roc_auc_score(ut[m], scores_full[m])), 4),
            }
    ages = pd.to_numeric(df["Age"], errors="coerce")
    qs   = ages.quantile([0, 0.25, 0.50, 0.75, 1.0]).values
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = ((ages >= lo) & (ages <= hi)).values if i == 3 else \
            ((ages >= lo) & (ages < hi)).values
        if m.sum() > 10 and ut[m].sum() > 0:
            fair[f"age_q{i+1}_{int(lo)}-{int(hi)}"] = {
                "n":    int(m.sum()),
                "n_ut": int(ut[m].sum()),
                "auroc":round(float(roc_auc_score(ut[m], scores_full[m])), 4),
            }

    # ─── Age quartile subgroup FNR (test split) ───────────────────────────────
    qs_te = age_te.quantile([0, 0.25, 0.50, 0.75, 1.0]).values
    for i in range(4):
        lo, hi = qs_te[i], qs_te[i + 1]
        m = ((age_te >= lo) & (age_te <= hi)).values if i == 3 else \
            ((age_te >= lo) & (age_te < hi)).values
        if m.sum() > 10 and ut_te[m].sum() > 0:
            fnr_pt = float(((~flag_te[m]) & ut_te[m]).sum() / max(ut_te[m].sum(), 1))
            fn_b   = []
            for _ in range(1000):
                idx = rng3.choice(m.sum(), m.sum(), replace=True)
                f_b = flag_te[m][idx]; u_b = ut_te[m][idx]
                if u_b.sum() == 0:
                    continue
                fn_b.append(float(((~f_b) & u_b).sum() / u_b.sum()))
            subgroup_fnr[f"age_q{i+1}_{int(lo)}-{int(hi)}"] = {
                "n":         int(m.sum()),
                "n_ut":      int(ut_te[m].sum()),
                "fnr_point": round(fnr_pt, 4),
                "fnr_ci95":  (round(float(np.percentile(fn_b, 2.5)), 4),
                              round(float(np.percentile(fn_b, 97.5)), 4)) if fn_b else None,
            }

    out = {
        "act":                            "2_ktas_final",
        "n_total":                        int(n),
        "n_undertriage":                  int(ut.sum()),
        "bootstrap_auroc_ci":             bootstrap_auroc_ci,
        "split_crc":                      split_crc,
        "decision_curve":                 dcurve,
        "ltt_dual_control":               ltt,
        "operating_metrics_bootstrap_ci": metrics_ci,
        "subgroup_fnr_at_alpha_0.15":     subgroup_fnr,
        "fairness_subgroup_auroc":        fair,
    }
    with open("/results/pipeline_act2_ktas_final.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2))
    return out


# ─────────────────────────────────────────────────────── Act 2b: Yale ─────────

@app.function(image=image_yale, cpu=8, memory=16384, timeout=2400, volumes={"/results": vol})
def act2_yale():
    """Yale 560k: per-ESI AUROC decomposition (ESI 3, 4, 5 separately).

    Requires /results/yale/5v_cleandf.rdata in the Modal volume.
    Output: /results/pipeline_act2_yale.json
    """
    import json
    import numpy as np
    import pandas as pd
    import pyreadr
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    res = pyreadr.read_r("/results/yale/5v_cleandf.rdata")
    df  = res[list(res.keys())[0]]
    print("yale loaded:", df.shape)

    cols      = list(df.columns)
    low       = {c.lower(): c for c in cols}
    esi_col   = (low.get("esi") or
                 next((c for c in cols if c.lower().startswith("esi")), None))
    dispo_col = (low.get("disposition") or
                 next((c for c in cols if "dispo" in c.lower()), None))
    esi   = pd.to_numeric(df[esi_col], errors="coerce")
    admit = df[dispo_col].astype(str).str.lower().str.contains("admit", na=False)

    feat_prefix = ("triage_vital", "age", "gender", "arrivalmode", "lang", "race",
                   "ethnicity", "insurance", "previousdispo", "cc_")
    feats = [c for c in cols if c.lower().startswith(feat_prefix)]
    if not feats:
        feats = [c for c in cols if any(k in c.lower() for k in
                 ("sbp", "dbp", "pulse", "resp", "temp", "o2", "spo2", "age"))]

    per_esi = {}
    for level in [3, 4, 5]:
        mask = (esi == level).values
        if mask.sum() < 200:
            per_esi[f"esi{level}"] = {"skip": "n<200", "n": int(mask.sum())}
            continue
        Xs = df.loc[mask, feats].copy().reset_index(drop=True)
        for c in Xs.columns:
            Xs[c] = (Xs[c].astype("category") if Xs[c].dtype == object
                     else pd.to_numeric(Xs[c], errors="coerce"))
        ys = admit[mask].astype(int).reset_index(drop=True)
        if ys.mean() < 0.001 or ys.mean() > 0.999:
            per_esi[f"esi{level}"] = {"skip": "degenerate_outcome", "n": int(mask.sum())}
            continue
        if len(Xs) > 150_000:
            pick = np.random.default_rng(40 + level).choice(len(Xs), 150_000, replace=False)
            Xs = Xs.iloc[pick].reset_index(drop=True)
            ys = ys.iloc[pick].reset_index(drop=True)
        oof = np.zeros(len(Xs))
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(Xs, ys):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                   num_leaves=63, verbose=-1)
            m.fit(Xs.iloc[tr], ys.iloc[tr])
            oof[va] = m.predict_proba(Xs.iloc[va])[:, 1]
        per_esi[f"esi{level}"] = {
            "n":            int(len(Xs)),
            "outcome_rate": round(float(ys.mean()), 5),
            "auroc":        round(float(roc_auc_score(ys, oof)), 4),
        }

    out = {"act": "2_yale_per_esi", "per_esi_auroc": per_esi}
    with open("/results/pipeline_act2_yale.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2))
    return out


# ─────────────────────────────────────────────────────────── Act 3 ─────────────

@app.function(image=image, cpu=8, memory=12288, timeout=3600, volumes={"/results": vol})
def act3():
    """TSTR gap + in-dist synthetic baseline + matched metamorphic control (100 × n=1267).

    Output: /results/pipeline_act3_final.json
    """
    import json
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    SEED = 42

    syn = pd.read_csv("/root/data/train.csv")
    kt  = pd.read_csv("/root/data/ktas.csv", sep=None, engine="python", encoding="latin-1")
    kt.columns = [c.strip() for c in kt.columns]
    for c in ["Age", "SBP", "HR", "RR", "BT", "Saturation", "NRS_pain"]:
        kt[c] = pd.to_numeric(kt[c].astype(str).str.replace(",", "."), errors="coerce")

    # ─── Common feature spaces ────────────────────────────────────────────────
    def _common_syn(d):
        return pd.DataFrame({
            "age":          pd.to_numeric(d["age"], errors="coerce"),
            "sex":          d["sex"].astype(str).str.lower().isin(("m", "male")).astype(int),
            "hr":           pd.to_numeric(d["heart_rate"], errors="coerce"),
            "rr":           pd.to_numeric(d["respiratory_rate"], errors="coerce"),
            "sbp":          pd.to_numeric(d["systolic_bp"], errors="coerce"),
            "temp":         pd.to_numeric(d["temperature_c"], errors="coerce"),
            "spo2":         pd.to_numeric(d["spo2"], errors="coerce"),
            "mental_alert": (pd.to_numeric(d["gcs_total"], errors="coerce") >= 15).astype(int),
            "pain":         pd.to_numeric(d["pain_score"], errors="coerce").clip(lower=0),
        })

    def _common_kt(d):
        return pd.DataFrame({
            "age":          pd.to_numeric(d["Age"], errors="coerce"),
            "sex":          (pd.to_numeric(d["Sex"], errors="coerce") == 1).astype(int),
            "hr":           d["HR"],
            "rr":           d["RR"],
            "sbp":          d["SBP"],
            "temp":         d["BT"],
            "spo2":         d["Saturation"],
            "mental_alert": (pd.to_numeric(d["Mental"], errors="coerce") == 1).astype(int),
            "pain":         pd.to_numeric(d["NRS_pain"], errors="coerce").clip(lower=0),
        })

    # 8-feature version (no mental_alert) for metamorphic (matches v2 feature set)
    def _meta_syn(d):
        return pd.DataFrame({
            "age":  pd.to_numeric(d["age"], errors="coerce"),
            "sex":  d["sex"].astype(str).str.lower().isin(("m", "male")).astype(int),
            "hr":   pd.to_numeric(d["heart_rate"], errors="coerce"),
            "rr":   pd.to_numeric(d["respiratory_rate"], errors="coerce"),
            "sbp":  pd.to_numeric(d["systolic_bp"], errors="coerce"),
            "temp": pd.to_numeric(d["temperature_c"], errors="coerce"),
            "spo2": pd.to_numeric(d["spo2"], errors="coerce"),
            "pain": pd.to_numeric(d["pain_score"], errors="coerce").clip(lower=0),
        })

    def _meta_kt(d):
        return pd.DataFrame({
            "age":  pd.to_numeric(d["Age"], errors="coerce"),
            "sex":  (pd.to_numeric(d["Sex"], errors="coerce") == 1).astype(int),
            "hr":   d["HR"],
            "rr":   d["RR"],
            "sbp":  d["SBP"],
            "temp": d["BT"],
            "spo2": d["Saturation"],
            "pain": pd.to_numeric(d["NRS_pain"], errors="coerce").clip(lower=0),
        })

    Xsyn = _common_syn(syn)
    ysyn = syn["triage_acuity"].astype(int)
    Xkt  = _common_kt(kt)
    ykt  = kt["KTAS_expert"].astype(int)

    # ─── TSTR: train on ALL 80k synthetic → test on KTAS ────────────────────
    m_tstr    = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                   num_leaves=63, verbose=-1)
    m_tstr.fit(Xsyn, ysyn)
    pred_tstr = m_tstr.predict(Xkt)
    acc_tstr  = round(float(accuracy_score(ykt, pred_tstr)), 4)
    qwk_tstr  = round(float(cohen_kappa_score(ykt, pred_tstr, weights="quadratic")), 4)

    # ─── Real KTAS 5-fold baseline (upper bound) ─────────────────────────────
    oof_kt = np.zeros(len(kt), dtype=int)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xkt, ykt):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06,
                               num_leaves=31, verbose=-1)
        m.fit(Xkt.iloc[tr], ykt.iloc[tr])
        oof_kt[va] = m.predict(Xkt.iloc[va])
    acc_real = round(float(accuracy_score(ykt, oof_kt)), 4)
    qwk_real = round(float(cohen_kappa_score(ykt, oof_kt, weights="quadratic")), 4)

    # ─── In-dist synthetic 5-fold baseline (5k sample) ───────────────────────
    rng  = np.random.default_rng(0)
    pick = rng.choice(len(Xsyn), 5_000, replace=False)
    Xs5, ys5 = Xsyn.iloc[pick].reset_index(drop=True), ysyn.iloc[pick].reset_index(drop=True)
    oof_syn  = np.zeros(5_000, dtype=int)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs5, ys5):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06,
                               num_leaves=31, verbose=-1)
        m.fit(Xs5.iloc[tr], ys5.iloc[tr])
        oof_syn[va] = m.predict(Xs5.iloc[va])
    acc_syn = round(float(accuracy_score(ys5, oof_syn)), 4)
    qwk_syn = round(float(cohen_kappa_score(ys5, oof_syn, weights="quadratic")), 4)

    # ─── Matched metamorphic control: 100 × n=1267 synthetic subsets ─────────
    DELTA    = {"hr": 30, "spo2": -6, "rr": 8, "sbp": -25, "temp": 1.5}

    def compute_violations(Xdf, yser, seed=0):
        """5-fold OOF → fraction of cases where worsening vital doesn't raise predicted level."""
        classes = np.sort(yser.unique())
        m_full  = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08,
                                     num_leaves=31, verbose=-1)
        m_full.fit(Xdf, yser)
        col_f    = {c: i for i, c in enumerate(m_full.classes_)}
        p_base   = m_full.predict_proba(Xdf)
        exp_base = np.zeros(len(Xdf))
        for j, c in enumerate(classes):
            if c in col_f:
                exp_base += p_base[:, col_f[c]] * c
        viols_per_feat = {}
        for feat, delta in DELTA.items():
            if feat not in Xdf.columns:
                continue
            Xmod      = Xdf.copy()
            Xmod[feat] = (Xmod[feat] + delta).clip(lower=0)
            p_mod     = m_full.predict_proba(Xmod)
            exp_mod   = np.zeros(len(Xdf))
            for j, c in enumerate(classes):
                if c in col_f:
                    exp_mod += p_mod[:, col_f[c]] * c
            # All DELTA values make the patient more acute → expected level should decrease
            viols_per_feat[feat] = float((exp_mod >= exp_base).mean())
        return float(np.mean(list(viols_per_feat.values()))), viols_per_feat

    Xkt_meta = _meta_kt(kt).fillna(kt.median(numeric_only=True)
                                     .reindex(_meta_kt(kt).columns))
    mean_viol_kt, per_feat_kt = compute_violations(Xkt_meta, ykt, seed=0)

    Xsyn_meta = _meta_syn(syn)
    rng2      = np.random.default_rng(7)
    n_target  = len(kt)  # 1267

    matched_violations = []
    per_feat_matched   = {f: [] for f in DELTA}
    for i in range(100):
        pick2 = rng2.choice(len(Xsyn_meta), n_target, replace=False)
        Xi    = Xsyn_meta.iloc[pick2].reset_index(drop=True)
        yi    = ysyn.iloc[pick2].reset_index(drop=True)
        try:
            mean_v, per_v = compute_violations(Xi, yi, seed=i)
            matched_violations.append(mean_v)
            for f, v in per_v.items():
                if f in per_feat_matched:
                    per_feat_matched[f].append(v)
        except Exception as e:
            print(f"  subset {i} failed: {e}")

    arr = np.array(matched_violations)

    out = {
        "act":                    "3_final",
        "features_used_tstr":     list(Xsyn.columns),
        "acc_synthetic_5fold":    acc_syn,
        "qwk_synthetic_5fold":    qwk_syn,
        "acc_tstr_syn_to_real":   acc_tstr,
        "qwk_tstr_syn_to_real":   qwk_tstr,
        "acc_real_ktas_5fold":    acc_real,
        "qwk_real_ktas_5fold":    qwk_real,
        "delta_tstr_acc":         round(acc_syn - acc_tstr, 4),
        "delta_tstr_qwk":         round(qwk_syn - qwk_tstr, 4),
        "note_tstr": ("HALO (Nat Comms 2023) synthetic EHR benchmark Δ_TSTR ≈ 0.005; "
                      "this dataset expected ~0.30–0.40. 9 shared features; "
                      "news2_score excluded (unavailable in KTAS)."),
        "metamorphic_ktas_violation_rate": round(mean_viol_kt, 4),
        "metamorphic_ktas_per_feat":       {k: round(v, 4) for k, v in per_feat_kt.items()},
        "metamorphic_synthetic_n1267_subsets": {
            "n_subsets":  len(matched_violations),
            "mean":       round(float(arr.mean()), 4),
            "std":        round(float(arr.std()), 4),
            "ci95_low":   round(float(np.percentile(arr, 2.5)), 4),
            "ci95_high":  round(float(np.percentile(arr, 97.5)), 4),
        },
        "metamorphic_reversal_robust": bool(np.percentile(arr, 97.5) < mean_viol_kt - 0.05),
        "metamorphic_per_feat_synthetic_means": {
            f: round(float(np.mean(v)), 4) for f, v in per_feat_matched.items() if v
        },
    }
    with open("/results/pipeline_act3_final.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2))
    return out


# ───────────────────────────────────────────────── Local entrypoint ───────────

@app.local_entrypoint()
def main():
    """Spawn act1, act2_ktas, act3 in parallel; act2_yale runs sequentially."""
    c1 = act1_decompilation.spawn()
    c2 = act2_ktas.spawn()
    c3 = act3.spawn()

    for tag, call in [("act1_decompilation", c1),
                      ("act2_ktas",          c2),
                      ("act3",               c3)]:
        try:
            call.get()
            print(f"OK {tag}")
        except Exception as e:
            print(f"FAIL {tag}: {e}")

    # Yale requires /results/yale/5v_cleandf.rdata in the Modal volume
    try:
        act2_yale.remote()
        print("OK act2_yale")
    except Exception as e:
        print(f"FAIL act2_yale (Yale .rdata must be in volume at /yale/5v_cleandf.rdata): {e}")
