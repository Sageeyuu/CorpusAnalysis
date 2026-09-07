#!/usr/bin/env python3
"""
词频统计：读 rows.csv 的 text 列 → 分词 → 统计频率 → 存 CSV

用法:
    python frequency.py                       # 默认读 data/rows.csv
    python frequency.py data/rows.csv --top 50
"""

import argparse
import re
from collections import Counter

import pandas as pd

DEFAULT_CSV = "data/rows.csv"

STOPWORDS = set("""
a an the and or but if of to in on for with at by from as is are was were
be been being this that these those it its he she they them his her their
you your i we our not no do does did will would can could should may might
""".split())

WORD_RE = re.compile(r"[a-zA-Z']+")
CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """英文按单词切，中文按单字切（没上 jieba 分词，先用最简单的办法）"""
    words = [w for w in WORD_RE.findall(text.lower())
             if len(w) > 1 and w not in STOPWORDS]
    chars = CJK_RE.findall(text)
    return words + chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV, help="rows.csv 路径")
    ap.add_argument("--top", type=int, default=50, help="终端里显示前 N 个高频词")
    ap.add_argument("--out", default="data/word_freq.csv", help="完整词频表输出路径")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)

    counter = Counter()
    for text in df["text"].dropna():
        counter.update(tokenize(text))

    if not counter:
        raise SystemExit("✗ 没数出任何词——text 列是空的，或者内容既不是英文也不是中文。")

    freq_df = pd.DataFrame(counter.most_common(), columns=["word", "count"])
    freq_df.to_csv(a.out, index=False, encoding="utf-8-sig")

    print(f"共 {len(freq_df):,} 个不同词/字，来自 {len(df):,} 行文字\n")
    print(f"前 {a.top} 高频：\n")
    for word, count in counter.most_common(a.top):
        print(f"  {count:>5}  {word}")
    print(f"\n✓ 完整词频表 → {a.out}")


if __name__ == "__main__":
    main()
