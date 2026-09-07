#!/usr/bin/env python3
"""
零膨胀检验：关键词在各文档里的出现次数，是不是比普通计数模型（Poisson/NB）
预期的 0 更多——即"零膨胀"（zero-inflated）。

原理：
    对每个关键词，把每一行/每一篇文档的出现次数当作因变量 y，
    文档长度（字数，标准化后）当作唯一协变量 x，分别拟合：
        Poisson(y ~ x)              NegativeBinomial(y ~ x)
        ZeroInflatedPoisson(y ~ x)  ZeroInflatedNegativeBinomial(y ~ x)
    用 AIC 比较拟合优劣，再用 Vuong (1989) 检验判断零膨胀模型是否显著更好
    （因为 Poisson vs ZIP 属于参数边界问题，标准似然比检验不适用，
    这也是文献里比较 count model 和 zero-inflated model 的标准做法）。

用法:
    python zero_inflation_test.py --keywords tariff,congestion
    python zero_inflation_test.py --keywords "关税,拥堵" --csv data/rows.csv

产出:
    data/zero_inflation_results.csv   每个关键词一行：n、零值占比、
                                       四个模型的 AIC、两个 Vuong 检验的 z/p 值、结论
"""

import argparse
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson

# ========================================================================
# 配置 CONFIG —— 要检验哪些关键词就改这里，或者用 --keywords 传参覆盖
# ========================================================================
DEFAULT_CSV = "data/rows.csv"
DEFAULT_KEYWORDS = ["tariff"]

WORD_RE = re.compile(r"[a-zA-Z']+")
CJK_RE = re.compile(r"[一-鿿]")


def text_length(text: str) -> int:
    """文档长度：英文按单词数，中文按字数，两者相加。跟 frequency.py 的分词口径保持一致。"""
    return len(WORD_RE.findall(text)) + len(CJK_RE.findall(text))


def keyword_count(text: str, keyword: str) -> int:
    """关键词在这篇文档里出现了几次。中文关键词按子串数，英文按整词边界数。"""
    if CJK_RE.search(keyword):
        return text.count(keyword)
    return len(re.findall(r"\b" + re.escape(keyword) + r"\b", text, re.I))


def vuong_test(ll_a: np.ndarray, ll_b: np.ndarray) -> tuple[float, float]:
    """Vuong (1989) 非嵌套模型比较。z>0 且显著 → 模型 a（零膨胀模型）更好。"""
    m = ll_a - ll_b
    n = len(m)
    z = np.sqrt(n) * m.mean() / m.std(ddof=1)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def _fit_one(model_cls, y, X, **kw):
    """拟合单个模型，失败或不收敛（AIC 非有限值）就返回 None 而不是让整个分析中断。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model_cls(y, X, **kw).fit(disp=0, method="bfgs", maxiter=500)
        if not np.isfinite(res.aic):
            return None
        return res
    except Exception:
        return None


def fit_models(y: np.ndarray, x: np.ndarray) -> dict:
    """标准化协变量后拟合四个模型。任何一个不收敛就跳过它，不影响其余模型。"""
    x_std = (x - x.mean()) / x.std()
    X = sm.add_constant(x_std)

    poisson = _fit_one(Poisson, y, X)
    nb = _fit_one(NegativeBinomial, y, X)
    zip_ = _fit_one(ZeroInflatedPoisson, y, X, exog_infl=X)
    zinb = _fit_one(ZeroInflatedNegativeBinomialP, y, X, exog_infl=X)

    out = {
        "poisson_aic": poisson.aic if poisson else None,
        "nb_aic": nb.aic if nb else None,
        "zip_aic": zip_.aic if zip_ else None,
        "zinb_aic": zinb.aic if zinb else None,
        "vuong_zip_vs_poisson_z": None, "vuong_zip_vs_poisson_p": None,
        "vuong_zinb_vs_nb_z": None, "vuong_zinb_vs_nb_p": None,
    }
    if poisson and zip_:
        z, p = vuong_test(zip_.model.loglikeobs(zip_.params), poisson.model.loglikeobs(poisson.params))
        out["vuong_zip_vs_poisson_z"], out["vuong_zip_vs_poisson_p"] = z, p
    if nb and zinb:
        z, p = vuong_test(zinb.model.loglikeobs(zinb.params), nb.model.loglikeobs(nb.params))
        out["vuong_zinb_vs_nb_z"], out["vuong_zinb_vs_nb_p"] = z, p
    return out


def analyze_keyword(df: pd.DataFrame, keyword: str) -> dict:
    counts = df["text"].apply(lambda t: keyword_count(t, keyword)).to_numpy()
    length = df["length"].to_numpy()
    n = len(counts)
    pct_zero = (counts == 0).mean()

    row = {"keyword": keyword, "n_docs": n, "pct_zero": round(pct_zero, 3),
           "mean_count": round(counts.mean(), 3)}

    if counts.sum() == 0:
        row["verdict"] = "全是 0，关键词从未出现，无法拟合模型"
        return row
    if n < 30 or (counts > 0).sum() < 5:
        row["verdict"] = f"样本太少（{n} 篇，{int((counts>0).sum())} 篇非零），结果不可靠，跳过"
        return row

    try:
        fits = fit_models(counts, length)
    except Exception as e:
        row["verdict"] = f"模型拟合失败: {type(e).__name__}"
        return row

    row.update({k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in fits.items()})

    def is_sig(z, p):
        return z is not None and p is not None and z > 1.96 and p < 0.05

    zip_ok = fits["vuong_zip_vs_poisson_z"] is not None
    zinb_ok = fits["vuong_zinb_vs_nb_z"] is not None
    zip_sig = is_sig(fits["vuong_zip_vs_poisson_z"], fits["vuong_zip_vs_poisson_p"])
    zinb_sig = is_sig(fits["vuong_zinb_vs_nb_z"], fits["vuong_zinb_vs_nb_p"])

    if not zip_ok and not zinb_ok:
        row["verdict"] = "模型未能稳定收敛，两个 Vuong 检验都跑不出来，结果不可靠"
    elif zip_sig or zinb_sig:
        note = "" if (zip_ok and zinb_ok) else "（NB/ZINB 未收敛，仅看 Poisson vs ZIP）" if not zinb_ok else ""
        row["verdict"] = f"零膨胀显著 (zero-inflated)：0 比 Poisson/NB 预期的更多，不只是文档短{note}"
    else:
        row["verdict"] = "无显著零膨胀：0 多主要是文档短/关键词本来就罕见，普通计数模型够用"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="rows.csv 路径")
    ap.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS),
                     help="逗号分隔的关键词列表，比如 tariff,congestion")
    ap.add_argument("--out", default="data/zero_inflation_results.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv).dropna(subset=["text"])
    df["length"] = df["text"].apply(text_length)
    keywords = [k.strip() for k in a.keywords.split(",") if k.strip()]

    print(f"读取 {len(df):,} 篇文档，检验 {len(keywords)} 个关键词：{keywords}\n")

    results = [analyze_keyword(df, kw) for kw in keywords]
    out_df = pd.DataFrame(results)
    out_df.to_csv(a.out, index=False, encoding="utf-8-sig")

    for r in results:
        print(f"[{r['keyword']}] n={r['n_docs']} 零值占比={r['pct_zero']:.1%}")
        print(f"  → {r['verdict']}\n")

    print(f"✓ 完整结果表 → {a.out}")


if __name__ == "__main__":
    main()
