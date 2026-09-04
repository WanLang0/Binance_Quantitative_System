"""从12个明细JSON重构 top20 macd div 汇总矩阵，补充详细字段（下单数/最大回撤/合计盈亏）。

在现有 top20_macd_div_summary_2024_2026.json 基础上，为每个配置逐年【新增】（保留原有字段）：
  n      = 该年各币种订单总数之和
  n_avg  = 该年单币平均订单数
  mdd_max= 该年最深单币回撤（用户关注的最大回撤）
并在配置级新增：
  tot        = 三年平均年收益连乘-1（合计盈亏）
  n_all      = 三年订单总和
  mdd_all_avg= 三年平均回撤(全部币种均值)
  mdd_all_max= 三年最深单币回撤
  sh_all     = 三年总夏普（三个年度夏普均值）
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')

SUMMARY = os.path.join(RES, 'top20_macd_div_summary_2024_2026.json')

TPL_ARG = {'不设': 'none', '只止损5%': 'sl5', '止盈止损各5%': 'tpsl5', '止盈止损5%': 'tpsl5'}
MODE_SUF = {'仅做多': 'longonly', '双向多空': 'longshort'}
YEARS = ['2024', '2025', '2026YTD']


def load_detail(tf, mode, tpsl):
    """按 (周期, 方向, 止盈止损) 定位明细文件并读取 results。"""
    if tpsl == '不设':
        # none 场景文件名无 tpsl 后缀；但 summary 里「仅做多/不设」实际用不带 longonly 的基准文件
        if mode == '双向多空':
            fname = f'top20_macd_div_{tf}_longshort_2024_2026.json'
        else:
            fname = f'top20_macd_div_{tf}_2024_2026.json'
    else:
        fname = f'top20_macd_div_{tf}_{MODE_SUF[mode]}_{TPL_ARG[tpsl]}_2024_2026.json'
    path = os.path.join(RES, fname)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d.get('results', {})


def _agg(yr_data, variant):
    """给定明细文件某年 {base: {variant: {...}}}，聚合出补充字段。"""
    mdds, ns = [], []
    for _t, row in yr_data.items():
        v = row.get(variant)
        if not v or v.get('ret') is None:
            continue
        if v.get('mdd') is not None:
            mdds.append(v['mdd'])
        if v.get('n') is not None:
            ns.append(v['n'])
    if not ns:
        return None
    return {
        'n': int(sum(ns)),
        'n_avg': round(sum(ns) / len(ns), 1),
        'mdd_max': round(min(mdds), 1) if mdds else 0.0,
    }


def main():
    with open(SUMMARY, encoding='utf-8') as f:
        summary = json.load(f)

    for strat in summary['strategies']:
        variant = strat['name']
        for cfg in strat['configs']:
            tf, mode, tpsl = cfg['tf'], cfg['mode'], cfg['tpsl']
            results = load_detail(tf, mode, tpsl)
            year_runs = []
            all_mdd_max = []
            all_n = 0
            for y in YEARS:
                yr = results.get(y)
                if not yr:
                    continue
                add = _agg(yr, variant)
                if not add:
                    continue
                cfg[y].update(add)   # 保留原有 ret/med/pos/cnt/mdd/sh/beat，新增 n/n_avg/mdd_max
                year_runs.append(cfg[y])
                all_n += add['n']
                all_mdd_max.append(add['mdd_max'])
            # 三年合计盈亏：平均年收益连乘-1
            if len(year_runs) == 3:
                tot = 1.0
                for en in year_runs:
                    tot *= (1 + en['ret'] / 100.0)
                cfg['tot'] = round((tot - 1) * 100, 1)
            else:
                cfg['tot'] = None
            cfg['n_all'] = int(all_n)
            cfg['mdd_all_avg'] = round(sum(en['mdd'] for en in year_runs) / len(year_runs), 1) if year_runs else 0.0
            cfg['mdd_all_max'] = round(min(all_mdd_max), 1) if all_mdd_max else 0.0
            # 三年总夏普：三个年度夏普（各自为跨币均值）再取平均
            shs = [en['sh'] for en in year_runs if en.get('sh') is not None]
            cfg['sh_all'] = round(sum(shs) / len(shs), 2) if shs else None

    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('重构完成 →', SUMMARY)


if __name__ == '__main__':
    main()
