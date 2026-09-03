# -*- coding: utf-8 -*-
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
d = json.load(open('scripts/results/us_all_backtest_1h_2025.json', encoding='utf-8'))
STRATS = ['KDJ','RSI','MACD','EMA','双均线','布林带','MACD基准','MACD+背离','背离+均线','背离+量能','背离+均线+量能']

# 每策略交易次数汇总
print("=== 各策略交易次数(114标的, 仅做多, 1h, 约608天=1.67年) ===")
print(f"{'策略':<14}{'平均笔数':>9}{'最少':>6}{'最多':>6}{'中位':>6}")
for s in STRATS:
    ns = []
    for tk, v in d.items():
        r = v['strat'].get(s)
        if r: ns.append(r['n'])
    ns = sorted(ns)
    avg = sum(ns)/len(ns)
    med = ns[len(ns)//2]
    print(f"{s:<14}{avg:>8.1f}{ns[0]:>6}{ns[-1]:>6}{med:>6}")

# 换算成频率
print("\n=== 交易频率换算(608天) ===")
for s in STRATS:
    ns = [v['strat'][s]['n'] for v in d.values() if s in v['strat']]
    if not ns: continue
    avg = sum(ns)/len(ns)
    print(f"{s:<14} 平均{avg:.1f}笔 ≈ 每{608/avg:.0f}天1笔")
