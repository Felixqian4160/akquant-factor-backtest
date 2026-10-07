"""V32: IC 排名 + 投票机制选股策略 (熊市 2022-2024)

机制 (完全因果, 无未来信息):
  ① 每 20 交易日调仓 (37 期)
  ② 每期用滚动 6 期 IC 对 412 因子排名 (只用已完成周期的 IC)
  ③ 选 |IC| 最大的前 N 个因子
  ④ 方向 = 滚动 IC 符号 (正 → 提名因子高分股 / 负 → 提名低分股), 各提名 top-10
  ⑤ 票数 >= M 的股票入选 (MIN=5 保底, MAX=20 封顶), 等权
  ⑥ 收益 = 入选股票 net20 均值 (T+1 open 买, T+21 close 卖, 扣 0.5% 成本), 37 期复利

对照:
  - 基准 (等权全股票)
  - 最强单因子 (talib_DX +95.8%)
  - 静态因子池 (H1 选池+方向 → H2 验证)

扫描: N ∈ {5,10,20} × M ∈ {2,3,4}
"""
import polars as pl
import pandas as pd
import numpy as np
from collections import Counter
import time, os, json, itertools

PANEL = "data/wavehunter_hs300_v31_refactored_v2_20260925.parquet"
NRUN = "evidence/v31_bear_factor_analysis_20260925/net20_cache.parquet"
OUTDIR = "evidence/v32_ic_voting_20260925"
W_S, W_E = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")
REBAL_STEP = 20
K_NOM = 10        # 每因子提名股票数
TRAIL = 6         # 滚动 IC 窗口 (期)
MIN_TRAIL = 3     # 最少历史期数
MAX_STOCKS = 20
MIN_STOCKS = 5

os.makedirs(OUTDIR, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("=" * 80)
log("V32: IC 排名 + 投票机制 (2022-2024)")
log("=" * 80)

# --- 1. net20 + rebal dates ---
net20 = pl.read_parquet(NRUN)
net20 = net20.filter(pl.col('trade_date').is_between(
    pl.lit(W_S.to_pydatetime()), pl.lit(W_E.to_pydatetime())))
all_dates = sorted(net20['trade_date'].unique().to_list())
rebal_dates = all_dates[::REBAL_STEP]
P = len(rebal_dates)
rebal_pd = [pd.Timestamp(d) for d in rebal_dates]
log(f"rebal dates: {P} ({rebal_pd[0].date()} → {rebal_pd[-1].date()})")

# net20 quick access: date_idx -> {ts_code: net20}
net20_at = {}
for p, d in enumerate(rebal_dates):
    sub = net20.filter(pl.col('trade_date') == d)
    net20_at[p] = dict(zip(sub['ts_code'].to_list(), sub['net20'].to_list()))
log(f"net20_at built: {sum(len(v) for v in net20_at.values())} rows")

# benchmark: equal weight
bm_rets = [np.nanmean(list(net20_at[p].values())) for p in range(P)]
bm_nav = np.concatenate(([1.0], np.cumprod([1+r for r in bm_rets])))
bm_total = bm_nav[-1] - 1
log(f"benchmark 3y: {bm_total*100:+.2f}%")

# --- 2. factor list ---
cols = pl.scan_parquet(PANEL).collect_schema().names()
SKIP = {'trade_date','ts_code','open','high','low','close','vol','amount'}
FACTORS = [c for c in cols if c not in SKIP]
log(f"factors: {len(FACTORS)}")

# --- 3. pass 1: IC + nominations per factor ---
log("Pass 1: computing IC matrix + nominations...")
t0 = time.time()
ic_matrix = {}   # factor -> np.array(P) (nan if no data/date)
nom = {}         # factor -> list of P tuples (hi_set, lo_set)
BATCH = 40
for bstart in range(0, len(FACTORS), BATCH):
    batch = FACTORS[bstart:bstart+BATCH]
    df = pl.read_parquet(PANEL, columns=['trade_date','ts_code'] + batch)
    df = df.filter(pl.col('trade_date').is_between(
        pl.lit(W_S.to_pydatetime()), pl.lit(W_E.to_pydatetime())))
    df = df.join(net20, on=['trade_date','ts_code'], how='left')
    for fac in batch:
        sub = df.select(['trade_date','ts_code',fac,'net20']).drop_nulls()
        if len(sub) < 100:
            ic_matrix[fac] = np.full(P, np.nan)
            nom[fac] = [(set(), set())] * P
            continue
        # IC per rebal date (rank corr)
        subr = sub.with_columns([
            pl.col(fac).rank().over('trade_date').alias('rf'),
            pl.col('net20').rank().over('trade_date').alias('rn'),
        ])
        ic_df = subr.group_by('trade_date').agg([
            pl.len().alias('n'),
            pl.corr('rf','rn').alias('ic'),
        ]).filter(pl.col('n') >= 20)
        ic_map = dict(zip(ic_df['trade_date'].to_list(), ic_df['ic'].to_list()))
        arr = np.array([ic_map.get(d, np.nan) for d in rebal_dates], dtype=float)

        # nominations: top-K desc / asc at each rebal date
        ranked = sub.with_columns([
            pl.col(fac).rank(descending=True).over('trade_date').alias('rk_hi'),
            pl.col(fac).rank().over('trade_date').alias('rk_lo'),
        ])
        hi = ranked.filter(pl.col('rk_hi') <= K_NOM).group_by('trade_date').agg(
            pl.col('ts_code').alias('codes'))
        lo = ranked.filter(pl.col('rk_lo') <= K_NOM).group_by('trade_date').agg(
            pl.col('ts_code').alias('codes'))
        hi_map = dict(zip(hi['trade_date'].to_list(), [set(c) for c in hi['codes'].to_list()]))
        lo_map = dict(zip(lo['trade_date'].to_list(), [set(c) for c in lo['codes'].to_list()]))
        nom[fac] = [(hi_map.get(d, set()), lo_map.get(d, set())) for d in rebal_dates]

        ic_matrix[fac] = arr
    log(f"  batch {bstart//BATCH+1}: {bstart+len(batch)}/{len(FACTORS)} done, {time.time()-t0:.0f}s")

ic_df_all = pd.DataFrame(ic_matrix).T  # factors x P
ic_df_all.to_csv(f"{OUTDIR}/ic_matrix_periods.csv")
log(f"IC matrix saved: {ic_df_all.shape}")

# --- 4. pass 2: voting backtest ---
def run_voting(N, M, K=K_NOM, trail=TRAIL, min_trail=MIN_TRAIL,
               max_stocks=MAX_STOCKS, min_stocks=MIN_STOCKS, pool=None, fixed_dir=None):
    """pool: 固定因子池 (None=滚动 IC 排名); fixed_dir: dict factor->'hi'/'lo' 固定方向"""
    rets = []
    picks_log = []
    for p in range(P):
        # 因子选择
        if pool is not None:
            sel_f = pool
        else:
            if p < min_trail:
                rets.append(0.0); picks_log.append([]); continue
            seg_start = max(0, p - trail)
            scores = {}
            for f in FACTORS:
                seg = ic_matrix[f][seg_start:p]
                seg = seg[np.isfinite(seg)]
                if len(seg) >= min_trail:
                    scores[f] = np.nanmean(seg)
            if not scores:
                rets.append(0.0); picks_log.append([]); continue
            sel_f = sorted(scores, key=lambda f: abs(scores[f]), reverse=True)[:N]

        # 投票
        votes = Counter()
        for f in sel_f:
            hi_set, lo_set = nom[f][p]
            if fixed_dir is not None:
                s = hi_set if fixed_dir[f] == 'hi' else lo_set
            else:
                seg_start = max(0, p - trail)
                seg = ic_matrix[f][seg_start:p]
                seg = seg[np.isfinite(seg)]
                if len(seg) == 0:
                    continue
                s = hi_set if np.nanmean(seg) >= 0 else lo_set
            for c in s:
                votes[c] += 1
        if not votes:
            rets.append(0.0); picks_log.append([]); continue
        sel = [c for c, v in votes.items() if v >= M]
        if len(sel) < min_stocks:
            sel = [c for c, _ in votes.most_common(min_stocks)]
        if len(sel) > max_stocks:
            sel = [c for c, _ in votes.most_common(max_stocks)]
        # 收益
        rv = [net20_at[p][c] for c in sel if c in net20_at[p] and np.isfinite(net20_at[p][c])]
        r = float(np.mean(rv)) if rv else 0.0
        rets.append(r)
        picks_log.append(sel)
    nav = np.concatenate(([1.0], np.cumprod([1 + r for r in rets])))
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    h1 = float(np.prod([1 + r for r in rets[:18]]) - 1)
    h2 = float(np.prod([1 + r for r in rets[18:]]) - 1)
    yr = {}
    for y in (2022, 2023, 2024):
        idx = [i for i in range(P) if rebal_pd[i].year == y]
        yr[y] = float(np.prod([1 + rets[i] for i in idx]) - 1)
    win = float(np.mean([r > 0 for r in rets if r != 0])) if any(r != 0 for r in rets) else 0
    avg_picks = float(np.mean([len(pk) for pk in picks_log if pk])) if any(picks_log) else 0
    return {'rets': rets, 'picks': picks_log, 'total': nav[-1] - 1, 'h1': h1, 'h2': h2,
            'mdd': mdd, 'yr': yr, 'win': win, 'avg_picks': avg_picks,
            'n_empty': sum(1 for pk in picks_log if len(pk) == 0)}

# 扫描 N x M
log("\nPass 2: voting sweep...")
rows = []
for N, M in itertools.product([5, 10, 20], [2, 3, 4]):
    r = run_voting(N, M)
    rows.append({
        'config': f'N{N}_M{M}', 'N': N, 'M': M,
        'total': r['total'], 'h1': r['h1'], 'h2': r['h2'],
        'y2022': r['yr'][2022], 'y2023': r['yr'][2023], 'y2024': r['yr'][2024],
        'mdd': r['mdd'], 'win': r['win'], 'avg_picks': r['avg_picks'], 'n_empty': r['n_empty'],
    })
    log(f"  N={N} M={M}: total={r['total']*100:+.1f}% H2={r['h2']*100:+.1f}% win={r['win']*100:.0f}% picks={r['avg_picks']:.1f}")

sweep = pd.DataFrame(rows).sort_values('h2', ascending=False)
sweep.to_csv(f"{OUTDIR}/voting_sweep_results.csv", index=False)

# 静态池对照: H1 选池 (从 split validation CSV 取 H1 收益 top N, 方向固定)
split = pd.read_csv("evidence/v31_bear_factor_analysis_20260925/bear_factor_split_validation.csv")
split_valid = split.dropna(subset=['ret_h1', 'ret_h2'])
for N_static in [10]:
    pool = split_valid.sort_values('ret_h1', ascending=False).head(N_static)
    pool_list = pool['factor'].tolist()
    fixed_dir = dict(zip(pool['factor'], pool['dir_h1']))
    r_s = run_voting(N=N_static, M=3, pool=pool_list, fixed_dir=fixed_dir)
    rows.append({
        'config': f'STATIC_H1top{N_static}_M3', 'N': N_static, 'M': 3,
        'total': r_s['total'], 'h1': r_s['h1'], 'h2': r_s['h2'],
        'y2022': r_s['yr'][2022], 'y2023': r_s['yr'][2023], 'y2024': r_s['yr'][2024],
        'mdd': r_s['mdd'], 'win': r_s['win'], 'avg_picks': r_s['avg_picks'], 'n_empty': r_s['n_empty'],
    })
    log(f"  STATIC (H1 top{N_static}): total={r_s['total']*100:+.1f}% H2={r_s['h2']*100:+.1f}%")

sweep = pd.DataFrame(rows).sort_values('h2', ascending=False)
sweep.to_csv(f"{OUTDIR}/voting_sweep_results.csv", index=False)

# 保存最佳配置的期度明细
best_cfg = sweep.iloc[0]
if best_cfg['config'].startswith('N'):
    N_b, M_b = int(best_cfg['N']), int(best_cfg['M'])
    r_best = run_voting(N_b, M_b)
else:
    r_best = run_voting(N=10, M=3, pool=pool_list, fixed_dir=fixed_dir)
periods_df = pd.DataFrame({
    'period': range(P), 'date': [d.date() for d in rebal_pd],
    'ret': r_best['rets'], 'n_picks': [len(pk) for pk in r_best['picks']],
})
periods_df.to_csv(f"{OUTDIR}/best_config_periods.csv", index=False)

# --- 5. 总结 ---
log("\n" + "=" * 80)
log("SUMMARY")
log("=" * 80)
log(f"benchmark: {bm_total*100:+.2f}%")
log(f"\nTop configs by H2 (半外验证):")
show = sweep.head(8)[['config','total','h1','h2','y2022','y2023','y2024','mdd','win','avg_picks']].copy()
for c in ['total','h1','h2','y2022','y2023','y2024','mdd']:
    show[c] = (show[c]*100).round(1)
show['win'] = (show['win']*100).round(0)
log(show.to_string(index=False))

log(f"\n最佳: {best_cfg['config']}")
log(f"  total={r_best['total']*100:+.1f}% H1={r_best['h1']*100:+.1f}% H2={r_best['h2']*100:+.1f}%")
log(f"  MDD={r_best['mdd']*100:.1f}% win={r_best['win']*100:.0f}% avg_picks={r_best['avg_picks']:.1f}")
log("=" * 80)
log("DONE")
