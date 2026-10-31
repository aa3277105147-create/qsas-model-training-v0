"""训练并评估 v0：预测 600 秒内跑完的概率 P 和成功时的总时间 T。

用法：python train_v0.py DATA_DIR OUT_DIR
DATA_DIR 中需包含 training-audited-600s.csv。
执行留一族、随机 5 折评估，再用全部有效数据训练并保存最终模型。
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

warnings.filterwarnings("ignore")

CANDS = ["c1", "c2", "c3", "c3b", "c4", "c5", "stim"]
NAMES = {"c1": "Aer 状态向量", "c2": "Aer MPS", "c3": "DDSIM", "c3b": "DDSIM hybrid",
         "c4": "Quimb MPS", "c5": "Qulacs", "stim": "Stim"}
TIMEOUT = 600.0
DENSE_MAX = 34            # 512 GiB 大内存档：c1/c5 只跑 n <= 34（数据里的事先判定）
INFERRED_WEIGHT = 0.5     # 推断不可行当负样本，但权重减半
FLAG_BELOW = 0.3          # 所有模拟器 P_ok 都低于它：标记“预计跑不完”
FEATS = ["log_n", "log_gates", "log_depth", "is_clifford", "log_cut", "gates_per_q", "depth_per_q", "cut_frac"]
MONO = {"log_n": 1, "log_gates": 1, "log_depth": 1, "log_cut": 1}   # 时间随这些单调不减；能否跑完单调不增
BANDS = [(2, 19, "2–19"), (20, 34, "20–34"), (35, 10**9, "35–1000")]
SEED = 0


def band(n):
    for lo, hi, name in BANDS:
        if lo <= n <= hi:
            return name
    return "?"


def load(data_dir: Path):
    a = pd.read_csv(data_dir / "training-audited-600s.csv", encoding="utf-8-sig")
    a = a[a.generation_status == "ok"].copy()
    circ = a.drop_duplicates("circuit_id").set_index("circuit_id")[
        ["family", "width", "gate_count", "depth", "is_clifford", "max_cross_cut_two_qubit_gates"]].copy()
    circ["is_clifford"] = circ.is_clifford.astype(str).str.lower().eq("true").astype(int)
    n = circ.width.astype(float)
    g = circ.gate_count.astype(float)
    d = circ.depth.astype(float)
    cut = circ.max_cross_cut_two_qubit_gates.astype(float)
    circ["log_n"] = np.log2(n)
    circ["log_gates"] = np.log10(g + 1)
    circ["log_depth"] = np.log10(d + 1)
    circ["log_cut"] = np.log10(cut + 1)
    circ["gates_per_q"] = np.log10(g / n + 1)
    circ["depth_per_q"] = np.log10(d / n + 1)
    circ["cut_frac"] = cut / (g + 1)
    circ["band"] = circ.width.map(band)

    # 每个（电路, 模拟器）的结果
    def outcome(s):
        if s == "ok":
            return "ok"
        if s in ("timeout", "out_of_memory"):
            return "fail"
        if s == "inferred_infeasible":
            return "fail_inferred"
        if s == "unsupported_precomputed":
            return "unsupported"
        return "unknown"          # error（1 条）
    a["outcome"] = a.status.map(outcome)
    lab = a.set_index(["circuit_id", "candidate_id"])[["outcome", "wall_s", "sim_s", "timing_clean", "stage"]]
    return circ, lab


def allowed(cand, row):
    if cand == "stim":
        return bool(row.is_clifford)
    if cand in ("c1", "c5"):
        return row.width <= DENSE_MAX
    return True


def lgb_params(kind):
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
             subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
             random_state=SEED, verbose=-1)
    sign = 1 if kind == "time" else -1
    p["monotone_constraints"] = [sign * MONO.get(f, 0) for f in FEATS]
    return p


class TwoStage:
    """每个模拟器：能否跑完 + 时间。"""

    def __init__(self):
        self.clf, self.reg, self.const_p = {}, {}, {}

    def fit(self, circ, lab, train_ids):
        for c in CANDS:
            rows = []
            for cid in train_ids:
                if (cid, c) not in lab.index:
                    continue
                r = lab.loc[(cid, c)]
                if r.outcome in ("ok", "fail", "fail_inferred"):
                    rows.append((cid, r.outcome, r.wall_s, r.timing_clean))
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["cid", "outcome", "wall_s", "clean"])
            X = circ.loc[df.cid, FEATS].values
            y = (df.outcome == "ok").astype(int).values
            w = np.where(df.outcome == "fail_inferred", INFERRED_WEIGHT, 1.0)
            if y.min() == y.max():
                self.const_p[c] = float(y[0])
            else:
                m = lgb.LGBMClassifier(**lgb_params("ok"))
                m.fit(X, y, sample_weight=w)
                self.clf[c] = m
            ok = df[df.outcome == "ok"]
            if len(ok) >= 5:
                m = lgb.LGBMRegressor(**lgb_params("time"))
                m.fit(circ.loc[ok.cid, FEATS].values, np.log10(ok.wall_s.astype(float).values))
                self.reg[c] = m
        return self

    def predict(self, circ, ids):
        X = circ.loc[ids, FEATS].values
        P, T = {}, {}
        for c in CANDS:
            if c in self.clf:
                P[c] = self.clf[c].predict_proba(X)[:, 1]
            else:
                P[c] = np.full(len(ids), self.const_p.get(c, 0.0))
            T[c] = 10 ** self.reg[c].predict(X) if c in self.reg else np.full(len(ids), TIMEOUT)
        return P, T

    def choose(self, circ, ids, flag_below=FLAG_BELOW):
        """按 T + (1 − P) × 600 选择；全部候选 P 低于阈值时标记低把握。"""
        P, T = self.predict(circ, ids)
        out, flags = [], []
        for i, cid in enumerate(ids):
            row = circ.loc[cid]
            ok = [c for c in CANDS if allowed(c, row)]
            flags.append(max(P[c][i] for c in ok) < flag_below)
            out.append(min(ok, key=lambda c: T[c][i] + (1 - P[c][i]) * TIMEOUT))
        return out, flags, P, T


def winner(lab, cid):
    best, bt = None, np.inf
    for c in CANDS:
        if (cid, c) in lab.index:
            r = lab.loc[(cid, c)]
            if r.outcome == "ok" and float(r.wall_s) < bt:
                best, bt = c, float(r.wall_s)
    return best


def best_time(lab, cid):
    w = winner(lab, cid)
    return None if w is None else float(lab.loc[(cid, w)].wall_s)


def cost(lab, cid, c):
    """选 c 的代价：跑完 = 总时间；跑不完、不允许或拒绝 = 600 秒（白等一个期限）。"""
    if c is None or (cid, c) not in lab.index:
        return TIMEOUT, False
    r = lab.loc[(cid, c)]
    if r.outcome == "ok":
        return float(r.wall_s), True
    return TIMEOUT, False


def evaluate(circ, lab, ids, choice, flags=None):
    rows = []
    flags = flags if flags is not None else [None] * len(ids)
    for cid, c, fl in zip(ids, choice, flags):
        bt = best_time(lab, cid)
        t, ok = cost(lab, cid, c)
        rows.append(dict(cid=cid, band=circ.loc[cid, "band"], family=circ.loc[cid, "family"], choice=c,
                         solvable=bt is not None, best=bt, t=t, ok=ok, winner=winner(lab, cid), flag=fl))
    df = pd.DataFrame(rows)
    return df


def summarize(df):
    s = df[df.solvable].copy()
    u = df[~df.solvable]
    s["t"] = s.t.astype(float)
    s["best"] = s.best.astype(float)
    s["ok"] = s.ok.astype(bool)
    ratio = s.t / s.best
    okr = ratio[s.ok]
    return dict(
        points=int(len(s)),
        top1=round(float((s.choice == s.winner).mean()), 3),
        within10=round(float((s.ok & (ratio <= 1.1)).mean()), 3),
        fail=int((~s.ok).sum()),
        fail_rate=round(float((~s.ok).mean()), 4),
        total_ratio=round(float(s.t.sum() / s.best.sum()), 3),
        geo_regret_ok=round(float(np.exp(np.log(okr).mean())), 3) if len(okr) else None,
        p90_regret_ok=round(float(np.quantile(okr, 0.9)), 2) if len(okr) else None,
        unsolvable=int(len(u)),
        reject_correct=int(u.choice.isna().sum()),
        flag_correct=int((u.flag == True).sum()),
        flag_false=int((s.flag == True).sum()),
        reject_false=int(s.choice.isna().sum()),
    )


def run_cv(circ, lab, splitter_name, eval_ids, all_ids):
    groups = circ.loc[all_ids, "family"].values
    if splitter_name == "lofo":
        folds = list(LeaveOneGroupOut().split(all_ids, groups=groups))
    else:
        folds = list(KFold(5, shuffle=True, random_state=SEED).split(all_ids))
    results = []
    probs = []
    eval_set = set(eval_ids)
    for tr_idx, te_idx in folds:
        tr = [all_ids[i] for i in tr_idx]
        te = [all_ids[i] for i in te_idx if all_ids[i] in eval_set]
        if not te:
            continue
        m = TwoStage().fit(circ, lab, tr)
        ch, fl, P, T = m.choose(circ, te)
        results.append(evaluate(circ, lab, te, ch, fl))
        for i, cid in enumerate(te):
            for c in CANDS:
                if (cid, c) in lab.index and lab.loc[(cid, c)].outcome in ("ok", "fail", "fail_inferred"):
                    r = lab.loc[(cid, c)]
                    probs.append(dict(cid=cid, cand=c, p=P[c][i], y=int(r.outcome == "ok"), src=r.outcome,
                                      t_pred=T[c][i], t=float(r.wall_s) if r.outcome == "ok" else np.nan,
                                      clean=str(r.timing_clean).lower() == "true"))
    return pd.concat(results, ignore_index=True), pd.DataFrame(probs)


def component_metrics(pr):
    out = {}
    for c in CANDS:
        d = pr[pr.cand == c]
        if d.empty:
            continue
        auc = roc_auc_score(d.y, d.p) if d.y.nunique() == 2 else None
        acc = float(((d.p >= 0.5).astype(int) == d.y).mean())
        ok = d[d.y == 1]
        err = np.abs(np.log10(ok.t_pred) - np.log10(ok.t))
        okc = ok[ok.clean]
        errc = np.abs(np.log10(okc.t_pred) - np.log10(okc.t))
        out[c] = dict(n=int(len(d)), n_fail=int((d.y == 0).sum()), auc=None if auc is None else round(auc, 3),
                      acc=round(acc, 3),
                      time_median_factor=round(float(10 ** err.median()), 2) if len(err) else None,
                      time_p90_factor=round(float(10 ** err.quantile(0.9)), 2) if len(err) else None,
                      time_median_factor_clean=round(float(10 ** errc.median()), 2) if len(errc) else None)
    return out


def family_breakdown(df):
    rows = []
    for fam, d in df[df.solvable].groupby("family"):
        rows.append(dict(family=fam, points=len(d), top1=round(float((d.choice == d.winner).mean()), 2),
                         fail=int((~d.ok.astype(bool)).sum()),
                         total_ratio=round(float(d.t.astype(float).sum() / d.best.astype(float).sum()), 2)))
    return pd.DataFrame(rows).sort_values("total_ratio", ascending=False)


def main():
    data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    circ, lab = load(data_dir)
    all_ids = list(circ.index)
    ncand = lab.reset_index().groupby("circuit_id").candidate_id.nunique()
    eval_ids = [c for c in all_ids if ncand.get(c, 0) == len(CANDS)]   # 补点电路只测了 1 个模拟器，不参与评估
    report = dict(circuits=len(all_ids), eval_circuits=len(eval_ids),
                  solvable=int(sum(best_time(lab, c) is not None for c in eval_ids)))
    for split in ["lofo", "random5"]:
        choices, probs = run_cv(circ, lab, split, eval_ids, all_ids)
        summ = summarize(choices)
        summ["by_band"] = {b: summarize(choices[choices.band == b]) for _, _, b in BANDS}
        report[split] = {"v0 两步模型": summ}
        choices.to_csv(out_dir / f"choices-{split}-v0.csv", index=False)
        report[split + "_components"] = component_metrics(probs)
        probs.to_csv(out_dir / f"predictions-{split}.csv", index=False)
        family_breakdown(choices).to_csv(out_dir / f"family-{split}.csv", index=False)
        # 模型选择的分布
        report[split + "_choice_counts"] = choices.choice.fillna("reject").value_counts().to_dict()
    # 用全部数据训练最终 v0，保存模型
    final = TwoStage().fit(circ, lab, all_ids)
    mdir = out_dir / "model-v0"
    mdir.mkdir(exist_ok=True)
    for c, m in final.clf.items():
        m.booster_.save_model(str(mdir / f"{c}-ok.txt"))
    for c, m in final.reg.items():
        m.booster_.save_model(str(mdir / f"{c}-time.txt"))
    (mdir / "meta.json").write_text(json.dumps(dict(
        features=FEATS, candidates=CANDS, dense_max=DENSE_MAX, timeout_s=TIMEOUT,
        const_p=final.const_p, monotone=MONO, inferred_weight=INFERRED_WEIGHT, flag_below=FLAG_BELOW,
        policy="argmin over allowed candidates of predicted_time + (1 - P_ok) * timeout_s",
        data="600s CPU baseline, training-audited-600s.csv (generation ok rows)",
        target="feasibility within 600 s; log10 total wall time (container start to exit)"), ensure_ascii=False, indent=1))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
