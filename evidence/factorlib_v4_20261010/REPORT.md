# factorlib 因子库 + fsdb v4 重算 — 审计报告

- 日期: 2026-10-09 / 2026-10-10
- 项目: `/media/felix/f/quant/akquant-factor-backtest`
- 类型: **基础设施与数据源验证（非 alpha 证据）**
- 状态: 完成 — factorlib 412 模块全量执行 PASS；v4 面板与锁定 v34 面板全量交叉核对完成

---

## 0. 本轮变更（2026-10-10）— 删除 2 个财务因子

**删除**: `accruals_sloan`、`gp_novymarx`
- **原因**: 依赖财报输入（bps / netprofit_yoy / grossprofit_margin）；stockdb 数据源不含财报；Tushare 不再作为数据源使用。除非将来接入新财报数据源，否则永远不可计算。
- **删除范围**: factorlib 模块文件 + 生成器映射 + 验证器计数 + 合同/manifest 记录 + v4 构建脚本排除逻辑（改为 `REMOVED_FACTORS` 文档记录）。
- **恢复路径**: 公式本体仍保留在 `examples/v34_build_part1_factors.py`（`f_accruals_sloan` / `f_gp_novymarx`）；未来若接入财报数据源，可恢复模块并重新加入生成器映射。
- **验证**: `bash factorlib_build/REGENERATE.sh` → 从公式源码重新生成 412 模块 + 全量真实执行 **412/412 PASS**（见 `regen_verify.txt`）；github 组 17 → 15，总数 414 → 412。

---

## 1. 交付物

### 1.1 factorlib — 独立因子计算库
- 位置: `factorlib/`；**412 个因子模块**（每因子一个 .py，统一入口 `compute(panel: pl.DataFrame) -> pl.Series`）
- 分组: alpha 107 / gtja 191 / talib 77 / academic 22 / github 15
- 来源: 从 aurumq_rl 公式源库（alpha101 / gtja191）与 v34 构建脚本 **逐函数 verbatim 提取**（AST），不依赖 aurumq_rl 运行时
- 一键重生成: `bash factorlib_build/REGENERATE.sh`（重建 + 全量执行验证）
- 校验和: `factorlib_sha256.txt`（全部模块 + manifest 的 sha256 清单）

### 1.2 fsdb v4 因子面板
- `data/wavehunter_hs300_fsdb_v4_20261009_235409.parquet`
- 1,388,089 行 × 551 列（fsdb v3 基础 139 列 + 重算因子 412 列），3.96 GB
- sha256 = `4d80e37ae70ea9f66fbdfdebd005ef8d3469f94c4ca868be98ce78ed089a11f1`
- 冻结输入契约: `v4_contract.json`（本合同副本已归档）
- 关键约定: 价格 `*_hfq × cum_latest`（还原上市锚定）；volume ÷100（手）；amount ÷1000（千元）；cap/float_mv ÷1e4（万元）；行业 = 申万二级；`p_pr` 用原始价（v34 语义）

### 1.3 验证/构建脚本（factorlib_build/）
| 脚本 | 用途 |
|---|---|
| `REGENERATE.sh` | 一键重生成 + 全量执行验证 |
| `verify_factorlib.py` | 412 模块逐一无跳真实执行 |
| `fidelity_check.py` | B 保真度：17 代表因子 v34 契约逐值比对 |
| `fidelity_low.py` | 61 因子重放（低相关 + 行业 alpha）+ 原始数据差异 |
| `replay_remaining.py` | 最终 3 因子重放（talib index / v_ep） |
| `input_rank_check.py` / `attrib_inputs.py` | 输入级归因 |
| `v4_build.py` | v4 构建（--init/--batch/--status/--assemble） |
| `cross_check_all.py` | 412 因子 × 30 日期 v4↔v34 交叉核对 |

---

## 2. 验证证据

### 2.1 保真度：与锁定 v34 面板逐值比对（EXACT 测试）
所有候选因子在 **v34 输入契约**上重放（1,298,335 行），与 v34 存储值逐值对比：

- 代表因子 17/17 **EXACT_PASS**（14 个精确 0.0；`r_tv` / `r_beta` / `p_m6` ≤ 5.5e-13 浮点噪声）
- 低相关候选重放（当前库，2026-10-10 重跑）:
  - Part 1: 43 因子中 **42 EXACT（0 差异）**；唯一例外 `p_pr` 为语义项（v34 = log 原始价；v4 已修正，修正后交叉 rho = 0.998）
  - Part 1b: 18 个行业中性 alpha **18/18 EXACT**
  - replay_remaining: `talib_MININDEX` / `talib_MAXINDEX` / `v_ep` **3/3 EXACT**
- **结论: 实现级无损**；全部 55 个 <0.90 低相关因子逐一重放 EXACT → v4 与 v34 的任何差异均来自输入数据源，而非公式实现

### 2.2 v4 ↔ v34 全量交叉核对（412 因子 × 30 抽样日期）
可比 405 / 412：

| rho 区间 | 因子数 |
|---|---:|
| ≥ 0.99 | 228 |
| 0.95 – 0.99 | 118 |
| 0.90 – 0.95 | 4 |
| 0.80 – 0.90 | 23 |
| < 0.80 | 32 |
| 无有效（见下） | 7 |

≥0.95 合计 **346 / 405 = 85.4%**。

7 个无法对比（均非实现问题）:
- `gtja_gtja_030`: 源库未完成 stub（需 Fama-French），双源全 NaN（行为一致）
- `talib_BBANDS_1` / `talib_MA` / `talib_MINMAXINDEX_0` / `talib_MINMAXINDEX_1`: v34 构建时按重复列剔除，v4 已产出
- `talib_MAX2` / `talib_MIN2`: v34 构建单参调用 bug（整列 NaN），v4 修复并产出真实值

### 2.3 低相关因子归因（55 个 < 0.90 = 35 alpha + 17 gtja + 2 talib + v_ep）
全部 55 个在 v34 契约上重放 EXACT；其差异来源（输入级实测）:

| 输入 | v4 vs v34 一致性 | 说明 |
|---|---|---|
| 原始 close | >1% 差异行 = 0.26% | 基本一致 |
| volume（手） | rank corr = 1.0000 | 换算后完全一致 |
| amount（千元） | rank corr = 1.0000 | 换算后完全一致 |
| 复权 close | 横截面 rank 0.980（min 0.966） | 两套复权库事件集不同（stockdb 事件多 7.7×；ratio 中位 0.935、p10 0.724）→ 秩敏感短窗口公式放大 |
| cap / circ_cap | 0.9993 | 一致 |
| pe | 0.806 | 口径不同 → `v_ep` = 0.81 |
| 行业 | — | 申万二级 vs Tushare 行业 → 18 个行业中性 alpha 的中性化基准偏移 |

### 2.4 关键工程修复（本轮）
1. **复权锚定**: `*_hfq × cum_latest` 还原上市锚定（level 因子 rank corr 0.67 → 0.99）
2. **`p_pr` 原始价语义**: 修正后 rho 0.683 → 0.998
3. **`talib_MAX2/MIN2` 单参 bug**: 修复（v34 全 NaN → v4 真实值）

---

## 3. 已知限制（诚实声明）
- 2 个财务因子已删除（见 §0）；恢复需新财报数据源
- 32 个 <0.80 因子是复权差异经秩敏感公式放大的产物 — 跨源对比时注意；v4 内部自洽
- `gtja_030` 双源均 NaN（源库 stub）
- 早期年份部分因子有效截面稀疏（如 alpha100 在 2000-2006 部分日期）
- **本报告为基础设施验证，不构成 alpha 证据**；不改变 v34-lb20 策略合同

---

## 4. 附件（本目录）
- `cross_check_all.csv` — 412 因子逐因子 rho
- `fidelity_low_rerun.txt` — 61 因子重放明细（2026-10-10 重跑）
- `replay_remaining.txt` — talib index / v_ep 重放
- `attrib_inputs.txt` — 输入级归因 + p_pr 语义 + 复权比例分布
- `regen_verify.txt` — REGENERATE.sh 运行记录（412/412 PASS）
- `factorlib_sha256.txt` / `v4_panel_sha256.txt`
- `v4_contract.json` / `v4_manifest.json` / `v4_factor_audit.csv`（构建契约与审计副本）

---

## 5. 复跑命令
```bash
cd /media/felix/f/quant/akquant-factor-backtest
export PYTHONPATH="/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages"
bash factorlib_build/REGENERATE.sh                 # 重建 + 验证 412 模块
/usr/bin/python3.12 factorlib_build/fidelity_low.py    # 61 因子 v34 契约重放
/usr/bin/python3.12 factorlib_build/replay_remaining.py
/usr/bin/python3.12 factorlib_build/cross_check_all.py # 412 × 30 日期交叉
/usr/bin/python3.12 factorlib_build/input_rank_check.py
```
