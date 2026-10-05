> Planning spec of the sentiment agent (written in Chinese, as used during the build). The module contract
> is `docs/CONTRACT.md` and the frozen test plan is `docs/PREREG.md`; where they differ from this
> plan, they are what the code does.

# B · Market Sentiment Agent — 方案 v0（研究版）

**本文是项目 2 的事实规划文档（SSOT）。** Agentic Trading / Market Sentiment（Track 2），与 A4（Alpha Factory / Open，`spec-A4-session-clientele.md`）并列为本届两个作品（D18）。截止 **2026-10-08 UTC+8**（D21；原 9/27）。
代码在 `sentiment/`（路径均相对项目目录 `b-sentiment-agent/`），数据在 `data/raw/news/` 等（gitignored；两个项目共用 monorepo 根目录的 `data/`，见 `sentiment/config.py`）。任务状态只看 `bd`。

一句话：**一个用 LLM 读市场情绪、在风控内自己决定 Bitget stock perp 仓位的 agent。它的 paper log 可以回测：同一套代码，历史时钟跑出回放日志，实时时钟在 Bitget Demo 上跑出实盘 paper 日志，两份日志格式相同、逐条对账。**

---

## 1. 提交结果由什么决定

| 项 | 内容 | 来源 |
|---|---|---|
| 打分 | **50% 量化**：paper trading 的 Sharpe、maxDD、胜率。**50% 评委**：决策可解释性、agent 架构、风控层有效性 | handbook Track 2 · `s2-rules.md` |
| 硬性材料 | 可运行 demo · 一条「事件 → 决策 → 执行」演示 · **比赛期间跑出的 paper log**（建议 ≥2 周） | 同上 |
| 有效性 | 合规 X 帖 · 表单六段 · 可访问材料；表单另有「LLM 的角色」字段 | handbook 第三章 |
| 格子 | Market Sentiment：9/23 普查公开 GitHub **0 个自报**；nocturne、aegis24、weekend-drift 可能改投（`operating-github-census.md`） | Lark Base |

**推论**：实时 paper log 只有 9/23–27 约 4 天，量化半边统计力很弱。能拉开差距的是：
1. 一份**可复现、做过泄漏审计的回放日志**（93 天）；
2. **LLM 相对固定规则基线的增量**；
3. 评委看得见的决策理由和风控拦截。

## 2. 核心设计：paper log 可回测

agent 是一个纯函数 `decide(state_t) → orders`，其中 `state_t` 只含 `available_at ≤ t` 的数据。时钟可以切换：

| 模式 | 时钟 | 执行 | 日志标签 |
|---|---|---|---|
| **replay** | 历史时间逐步推进 | 成交模拟器（下一根 1h bar 开盘价 + 半价差 + taker + 实际 funding） | estimated |
| **live** | 现在 | Bitget Demo 下单（`paptrading: 1`） | observed |

两种模式共用：
- 数据层（point-in-time 取数）
- prompt
- LLM 缓存（`prompt 哈希 → 回复`；重跑回放时完全确定，评委可离线复现）
- 风控层
- 日志格式（append-only JSONL + 哈希链）
- 指标代码

**回放与实盘对账**：live 每跑完一小时，就对同一小时补跑 replay。决策应完全一致（同一份缓存）；再比较 Demo 实际成交和模拟成交差多少 bp。这是回放可信度的主证据。

## 3. 数据（2026-09-23 实测）

| 输入 | 来源 | 覆盖 | point-in-time 键 |
|---|---|---|---|
| 新闻 | MCP `news_label_search`（label 1/2/6/7/9） | 6/22–9/23 共 **622 条**（去重）；每天中位数 4 条，26 个空白日；正文中位数 5.3k 字；75% 提到宇宙内代码（NVDA 269 条、AAPL 122 条……）；9/17 起新增「Market Movement Analysis」系列，单日 151 条 | `published_at`（毫秒） |
| 分析师评级 / 目标价 | MCP `equity_estimates_price_target` | NVDA 一只 972 条，回溯到 2016；窗口内每月约 1–30 条 | `rating_date`，**次一交易时段起**才可见 |
| 内部人交易 | MCP `equity_ownership_insider_trading` | 稀疏（TSLA 窗口内约 5 条） | `filing_date` |
| 加密恐慌贪婪指数 | MCP `crypto_sentiment_crypto_fear_greed` | 日度，长历史 | 次日才可见 |
| 市场恐慌贪婪指数 | MCP `sentiment_market_fear_greed` | **只有当前值**（加前收、一周前、一月前、一年前） | 只用于 live |
| 资金费率 | 本地 `data/raw/funding/` | 从 2026-06-21 起（约 90 天） | 结算时间 |
| stock perp 1h K 线、rToken 现货 | 本地 `data/raw/perp/`、`spot/` | 从 2025-08 起 | bar 收盘时间 |
| 多空账户比、多空持仓比、主动买卖量 | Bitget v2 | **stock perp 全部为空**（只有 crypto 有，约 4 天） | 不可用 |

**Qwen 知识截止点探针（R1，`sentiment/probes/cutoff_probe.py` → `out/cutoff_probe.json`）**：按月闭卷问 2024-07 到 2026-08 的月末收盘价和月度涨跌。
- 月末收盘价的误差：到 2025-04 为止约 2–4%；2025-05 升到 6%，2025-06 为 11%；之后在 17–45% 之间。
- 月度涨跌方向：2025-11 起回到约 50%（瞎猜水平）。
- 模型自报的截止点是「2025 年初」。

**结论：截止点约为 2025-04/05。回放窗口 2026-06-22→09-22 比它晚 13 个月以上，模型靠参数记忆无法作答**（status: observed）。模型如果在服务端更新，要重跑探针。

## 4. 宇宙

取 Demo 盘可交易 ∩ 新闻覆盖 ∩ 本地有完整数据的交集：

- **主宇宙**：**NVDA, AAPL, GOOGL, META, AMZN, TSLA, MSTR, COIN, HOOD, CRCL**（10 只，perp 从 2025-08 起、funding 从 6/21 起）。
- **待定**：SP500USDT、NDX100USDT，拟作为市场腿对冲，本地尚无数据，需要另外抓取。
- **不纳入**：SPCX、SNDK，本地历史过短。

## 5. Agent 循环

1. **证据抽取（LLM 阶段 1，按条缓存）**：每条新闻、评级、内部人交易到达时，抽成一张证据卡，字段为：涉及的代码、立场（−2..+2）、强度、时效、是否新信息、原文引用。一条只抽一次，在 replay 和 live 之间复用。这也是控制 token 成本的关键。
2. **决策（LLM 阶段 2）**：每 4 小时一次，另外在有新证据卡时触发。输入是：
   - 最近的证据卡；
   - 每只标的的价格、溢价、funding 状态；
   - 恐慌贪婪指数所处的阶段；
   - 当前持仓。

   输出 JSON：每只标的的目标权重 ∈ [−w, w]、理由、引用了哪些证据卡 id、置信度。
3. **风控层（确定性）**：
   - 单名上限、总敞口上限；
   - 净敞口（beta）限制；
   - 止损和日内亏损熔断；
   - 美股休市期间降低上限（流动性差）；
   - funding 异常时禁止开仓；
   - LLM 输出不合法时维持原仓。

   每次拦截都写进日志。
4. **执行**：把目标权重和当前持仓的差值换算成订单，走 replay 的模拟器或 live 的 Demo。

## 6. 对照组与泄漏审计（全部在同一个回放里跑）

| 对照 | 作用 |
|---|---|
| **固定规则基线** | 证据卡立场按规则求和再定仓位（不经过 LLM 决策）；再加 funding 拥挤、恐慌贪婪指数阶段。**LLM 增量 = LLM 版 − 基线**，用分块 bootstrap 给置信区间。9/23 快查：单独的 funding 因子 RankIC 为 +0.02（t=1.35，not significant），基线本身预计没有边际 |
| 不给新闻 | 只喂价格和 funding。如果它也能赚钱，说明收益来自动量或泄漏，而不是情绪 |
| 打乱新闻 | 把证据卡随机换到错误的日期。效果如果还在，说明收益和新闻无关 |
| 脱敏 vs 原文 | 代码换成化名、去掉日期。两者差距大，说明模型在用名字作先验 |
| 事后解释类新闻 | 「Market Movement Analysis」描述的是已经发生的涨跌，只能在 `published_at` 之后可见；另外单独报告去掉这一系列后的结果 |

## 7. 窗口与指标

- **回放**：2026-06-22 → 09-22。
  - **开发段** 6/22–8/10：调 prompt 和风控参数，每次改动记一条 [`trials.log`](../trials.log)。
  - **检验段** 8/11–9/22：不调参。
- **实盘 paper**：Demo 上线时起到 9/27（第一段，已封存 9/23–9/28）；截止延到 10/8 后可续跑第二段到 10/7。
- **指标**：Sharpe、Sortino、maxDD、胜率、换手率、费用和滑点（按阶梯逐级报）、LLM 相对基线的增量及置信区间、回放与实盘对账的 bp 差。每个数字标 observed / estimated。

## 8. 执行与日志

- **首选**：Bitget Demo（`paptrading: 1`，USDT-FUTURES）。9/23 实测：demo 盘有 45 个合约，包含上面 10 只标的以及 SP500、NDX100，报价跟着实盘走（NVDA 229.22 对 229.49）。**需要用户在 Bitget Demo Trading 创建 API key**，写入 `.env` 的 `BITGET_DEMO_*`。
- **备选**：官方路径 GetAgent Skill → Playbook → GetAgent Studio Paper Trading，或 Agent Hub `--paper-trading`。
- **兜底**：影子账本，按实盘报价记账，不需要 key，可信度最低。
- **日志**：每轮一行，字段为：`ts, mode, input_snapshot_hash, evidence_ids, llm_request_hash, llm_output, baseline_output, risk_actions, orders, fills, position, equity, prev_hash`。每小时公开发布一次。

## 9. 成本预算

- **证据抽取**：约 622 条新闻加评级，每条 1–3k token，一次性约 1.5M token，结果缓存。
- **决策**：约 560 轮 × 约 3k token，每跑一遍约 1.7M token。
- Qwen 额度为 30U，按网关实际计费核对（尚未确认单价）。**每次改 prompt 等于重跑一遍，prompt 改动次数要控制。**

## 10. 日程（UTC+8）

| 日期 | 做什么 |
|---|---|
| 9/23 晚 | 方案（本文）；数据抓取器；知识截止点探针 ✅ |
| 9/24 | 数据层（point-in-time 取数）+ 证据抽取 + 成交模拟器 + 日志格式；拿到 Demo key 就让 live 上线 |
| 9/25 | 决策 prompt 和风控；开发段回放；基线和对照组 |
| 9/26 | 检验段回放（只跑一次）；对账；报告和 demo 页面；周末 live 继续 |
| 9/27 | README、表单、X 帖、≤3 分钟视频；截止前停止 live 并封存日志 |

## 11. 风险与 No-Go

- **拿不到 Demo key**：live 改用影子账本，表单里如实写明。
- **检验段 LLM 增量 ≤ 基线**：照实提交，把「情绪 agent 在这个窗口没有跑赢规则」写进摘要。评委那一半分看的是架构、可解释性和风控，不是收益。
- **Qwen 额度或接口不稳**：缓存优先；决策层可以改用更小的模型，但要标明。
- **A4 产能冲突**：A4 已进入报告和站点收尾阶段。B 不改动 A4（`a4-session-factor/`）的 `src/`、`judge.sh`、`report/`，也不 import A4 的代码。

