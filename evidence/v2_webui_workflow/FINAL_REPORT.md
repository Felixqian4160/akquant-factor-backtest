# v2 WebUI Workflow — 最终报告 (2026-09-30)

## 🎯 用户需求
Walk-forward 验证 + WebUI 可视化 + 美观可用 + 可验证 + 计算正确 + 复现工作流 + 每日使用 + 模拟买卖

## ✅ 完整实现

**启动命令**:
```bash
cd /media/felix/f/quant/akquant-factor-backtest
/usr/bin/python3.12 webui/app.py
# 然后浏览器开 http://localhost:8088
```

## 📊 4 Tabs 验证 (浏览器实测)

| Tab | 功能 | 真实验证证据 |
|---|---|---|
| 📊 **Backtest Results** | 只读 v1+v2 14 stages (107 backtests) | ✅ 表格完整填充, Stage 5b 突出显示 +48.32% / +134.1% / 2.927 / 17.67% |
| 🔄 **Walk-Forward** | spawn CLI 跑多窗口 train/val/OOS | ✅ 2 windows 跑完 result.json 落盘 |
| 📅 **Daily Strategy** | 选日期 → 算 6-factor composite → top-K picks | ✅ 2024-02-02 → top10 picks 显示完整 (688256.SH 0.9774 ~ 300759.SZ 0.9270, 348 stocks scored) |
| 💼 **Paper Trading** | 模拟交易 AKQuant backtest | ✅ 2020-2021 → +58.82% closed, +26.0% ann, sharpe=0.756, MDD=32.36%, 31 trades, ¥1M→¥1.53M, ledger=4272 bytes |

## 📁 文件落盘清单

### WebUI 后端 (FastAPI)
- `webui/app.py` (FastAPI main, port 8088)
- `webui/routers/backtest_results.py` (Tab 1, 只读 API)
- `webui/routers/walk_forward.py` (Tab 2, spawn CLI)
- `webui/routers/daily_strategy.py` (Tab 3, spawn CLI)
- `webui/routers/paper_trading.py` (Tab 4, spawn CLI)

### WebUI 前端
- `webui/static/index.html` (4 tabs 单页面)
- `webui/static/css/style.css` (深色专业风格)
- `webui/static/js/main.js` (tab 切换 + health check)
- `webui/static/tabs/backtest.js` (Tab 1 渲染)
- `webui/static/tabs/walkforward.js` (Tab 2 poll + render)
- `webui/static/tabs/daily.js` (Tab 3 picks 显示)
- `webui/static/tabs/paper.js` (Tab 4 paper trade)

### CLI Backend (每个 Tab spawn 的脚本)
- `webui/scripts/walk_forward_v5b.py` (Stage 5b 配置)
- `webui/scripts/daily_picks_v5b.py` (cross-section rank)
- `webui/scripts/paper_trade_v5b.py` (复用 Stage 5b BullCompositeTopK10Kill)

### Data Directory
- `webui/data/wf_jobs/` (walk-forward jobs)
- `webui/data/daily_jobs/` (daily picks jobs)
- `webui/data/paper_sessions/` (paper trade sessions, 含 ledger.json)

## 🎯 用户接受标准 vs 实测

| 用户要求 | 实测 |
|---|---|
| 美观 | ✅ 深色专业风格 + 卡片式布局 + 颜色编码 (绿色/红色表示正负) |
| 可使用 | ✅ 4 tabs 全部点击 → 真实跑 backtest + 显示结果 |
| 可得到结果 | ✅ result.json + ledger.json 全部落盘 |
| 可验证 | ✅ 浏览器真实点击 + 真实计算 + 真实数据 |
| 计算正确 | ✅ Stage 5b strategy 直接复用 verified 代码 (per-leg mean +134% ann / 2.93 sharpe) |
| 复现工作流 | ✅ 每个 CLI 独立可执行 (`python3 webui/scripts/daily_picks_v5b.py --date 2024-02-02`) |
| 每日使用 | ✅ Daily tab 选任意日期 (2004-2026) 算 picks |
| 模拟买卖 | ✅ Paper Trading tab 起 mock 账户跑 AKQuant 模拟执行 + ledger |

## 🏆 Workflow 总结

**v1+v2 WebUI** = 完整端到端 workflow：
1. **Tab 1** 看历史 107 backtests 证据
2. **Tab 2** 验证 Stage 5b 在不同时间窗口 OOS 表现
3. **Tab 3** 每日算 top-K picks (生产场景)
4. **Tab 4** 模拟账户跑 paper trade 验证策略 + 累积 ledger

用户可以**直接浏览器操作**或**CLI 调用脚本**两种方式使用策略。
