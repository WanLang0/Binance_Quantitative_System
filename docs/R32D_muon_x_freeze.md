# R32-D / Muon-X — Observation v1（研究封账）

**日期**：2026-09-20
**状态**：`Research Candidate — Observation / No Execution`
**冻结范围**：策略代码、观察器代码、参数、信号口径全部冻结。此后仅允许修 bug 与数据完整性修复。

---

## 1. 身份与定位

Muon-X（R32-D）**不是 Alpha**，是 Muon 空窗期的 Beta 风险转换器：用部分回撤风险换取空窗期潜在收益。长期不保证超额收益，主要改变收益分布形状。

```
Muon Gate（Crypto 横截面环境）
   ├─ ON  → Muon Alpha（U8 PIT + R14 + ExpRank + MA60 gate，Production）
   └─ OFF → Nasdaq Gate（MA50 > MA200）
              ├─ YES → QQQ Beta
              └─ NO  → Cash（真实基准 4.5%/y，非 0）
```

口径：T 日美股收盘算信号 → T+1 生效；OFF-only overlay，ON 期零污染。

## 2. 证据账本

| 实验 | 结论 |
|---|---|
| R32 样本内（2024-01→2026-08） | D +190.7%/Sh 1.31/MDD -30.3%（MDD 零恶化）；B Always QQQ 淘汰（2022 压力 -48.1% MDD）；E 五态淘汰（无增量）。*数字为 2026-09-20 未来函数审计修正后（gate 持仓 shift(1)），修正前 D=+188.4%，结论不变* |
| R33-1 真历史（2007→2026） | D 腿全史 CAGR 13.2%/MDD -28.6% vs QQQ BH 15.2%/-49.5%——**跑不赢 BH，价值是折减回撤**。弱点暴露：2020 式金叉内崩盘照单全收（-28.6%）、2025 式震荡 whipsaw 白干（+1.8% vs BH +20.2%） |
| R33-2 参数邻域 | (20,100)/(50,200)/(100,200)/(50,250)/(50,300) CAGR 全落 10~13% 窄带 → **plateau 确认，regime 效应，非参数特效** |
| R33-3 现金修正 | Cash 4.5%/y 使 A 底线 +107.0%→+122.1%；D 几乎无感（OFF 期已持仓 96%）。**R34 的比较基准必须是 Cash，不是 0** |

历史基线：`scripts/tmp_r33_d_validation.py`（2007-2026 逐年 regime 表，只读先验，不混入 live OOS）。

## 3. 观察器（已冻结）

`scripts/qqq_observer.py` → `data/xasset_observer.jsonl`（每次 Muon 调仓追加一条）。

设计纪律：观察≠执行 / 软依赖 / 异步静默 / QQQ 本地增量缓存 / 不补种历史记录（R33 只作先验）。

## 4. R34 — Live/OOS Audit（固定审计表，届时勿增删）

| 项目 | 指标 | 判断 |
|---|---|---|
| 空窗效率 | **D - Cash**（OFF 期累计，Cash 按 4.5%/y） | 是否产生**净**增益 |
| 风险代价 | D MDD vs Cash MDD | 增加多少回撤 |
| Whipsaw | 翻转次数 × 50bps 往返 | 是否显著 |
| 趋势捕获 | D vs QQQ BH（OFF 期） | 捕获比例 |
| 同步风险 | Muon DD × QQQ DD 相关 | 是否危险重叠 |
| 极端状态 | `overlap_dd_risk` 次数/持续 | — |
| 稳定性 | 不同 OFF regime 是否重复出现 | — |

**裁决只有三种**：PASS（→ Experimental）/ HOLD（继续观察）/ FAIL（放弃）。
判据锚点：不看收益高低，只看 **D - Cash 是否足以补偿新增尾部风险**。

## 5. 生命周期

```
R33 Historical（先验） → R32-D Research → Live/OOS Observation → R34 Audit
                                                              → Experimental / HOLD / FAIL
```

R35/R36 及以后：不再挖参数。接下来最有价值的数据来自市场本身。

---
*Moon 系列状态：Muon = Production ✅ ｜ Muon-X = Research-Observation ⭐ ｜ B/E = 淘汰 ❌ ｜ C = 对照组保留*
