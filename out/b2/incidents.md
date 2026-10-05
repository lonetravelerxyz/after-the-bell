# Incident-exit pilot (estimated, Bitget spot candles)

Events: DefiLlama hacks >= $20M since 2021-01-01: 112 listed, 63 with a plausible token, 14 with Bitget 1-minute data and a >= 5% trailing-60-minute drop on day 0 or day 1 (the alert proxy). Survivorship: tokens delisted from Bitget since (many of the worst outcomes) return error 400 and are missing, so the bleed below is understated.

| Window after the alert | Median token return | Mean | Share < -10% | Median BTC-adjusted |
|---|---:|---:|---:|---:|
| +1h | -0.4% | -2.7% | 7% | -0.5% |
| +6h | -4.9% | -6.9% | 29% | -4.2% |
| +24h | -9.7% | -15.3% | 46% | -8.7% |
| +72h | -12.0% | -19.3% | 54% | -13.6% |
| +7d | -12.5% | -14.9% | 62% | -14.7% |
| +30d | -32.8% | -35.1% | 90% | -34.1% |
| 30-day low | -38.9% | -37.1% | 86% | |

Unavoidable part (day-0 00:00 UTC to the alert minute): median -5.0%, mean -3.8%. Pre-alert 60-minute drop at the alert: median -5.7%.

| Date | Event | Token | Loss | Alert (UTC) | Unavoidable | +1h | +24h | +7d | +30d | 30d low |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 2021-09-29 | Compound V2 | COMP | $147M | 09-29 23:13 | -5.6% | -1.0% | +2.6% | +1.8% | - | -3.9% |
| 2022-11-12 | FTX | FTT | $450M | 11-12 00:36 | -6.4% | -2.8% | -16.3% | -39.5% | -38.1% | -49.2% |
| 2023-07-30 | Curve DEX | CRV | $62M | 07-30 16:22 | -5.7% | +0.3% | -11.0% | -11.1% | -29.1% | -36.5% |
| 2023-11-22 | KyberSwap Elastic | KNC | $48M | 11-22 23:08 | +1.7% | +2.4% | +1.3% | -2.0% | - | -11.4% |
| 2024-05-20 | Gala | GALA | $22M | 05-20 20:22 | +3.1% | -9.9% | -2.5% | +2.7% | -36.5% | -41.2% |
| 2024-10-16 | Radiant V2 | RDNT | $50M | 10-16 17:49 | -8.1% | +0.4% | -5.5% | -14.4% | -20.2% | -42.1% |
| 2025-05-22 | Cetus CLMM | CETUS | $223M | 05-22 10:41 | +5.6% | -24.9% | -24.9% | -28.8% | -60.3% | -63.8% |
| 2025-07-09 | GMX V1 Perps | GMX | $42M | 07-09 13:17 | -4.3% | -9.7% | -15.0% | -3.4% | +7.2% | -22.0% |
| 2025-11-03 | Balancer V2 | BAL | $128M | 11-03 10:03 | -7.2% | -2.2% | -9.3% | -4.2% | -26.1% | -32.1% |
| 2026-03-21 | Resolv USR | RESOLV | $24M | 03-22 02:41 | -9.0% | +6.7% | - | -31.7% | - | -43.0% |
| 2026-04-01 | Drift Trade | DRIFT | $295M | 04-01 17:18 | -1.9% | -7.6% | -31.2% | -36.7% | -36.8% | -60.2% |
| 2026-04-18 | Kelp | KERNEL | $293M | 04-18 18:41 | -7.8% | +4.6% | -9.7% | -12.5% | -20.5% | -21.9% |
| 2026-06-08 | Humanity | H | $32M | 06-08 08:04 | -3.9% | +4.9% | -79.0% | -14.4% | -90.3% | -92.6% |
| 2026-09-24 | Bitget | BGB | $387M | 09-24 20:51 | -4.2% | +0.6% | +1.8% | - | - | +0.1% |

Skipped (90): PlayDapp (no token); FixedFloat (error 400); CoinsPaid (no token); Uranium Finance (no token); MonoX (error 400); Stake.com (no token); ThalaSwap (error 400); Munchables (no token); 3Commas (error 400); COLDCARD (no token); MEV-Boost Relay (no 1min data (0 rows)); Ooki (error 400); UwU Lend (no token); Lykke (no token); Mango Markets V3 (error 400); Liquid Network (no token); Poloniex (no token); Penpie (error 400); StakeHound / Fireblocks (no 1min data (0 rows)); Bitget (no 5%/60min drop on day0-1); Beanstalk (error 400); Qubit (error 400); Badger DAO (error 400); Paid Network (no token); Rari Capital (error 400); DMM Bitcoin (no token); BtcTurk (no token); BonkDAO (no token); Heco Bridge (error 400); AlphaPo (no 5%/60min drop on day0-1); Orbit Bridge (error 400); Mirror (error 400); Crypto.com (no token); Vulcan Forged (no token); Harmony Bridge (no token); Bybit (no token); Portal (no token); Grim Finance (error 400); Atomic Wallet (no token); Meerkat Finance (no token); Hedgey (error 400); AnubisDAO (no token); Liquid Global (no 1min data (0 rows)); Multichain (error 400); Truebit (no token); BigONE (no token); Spartan (error 400); Elrond (no token); US Government Crypto Wallet (no token); Binance Bridge (no 5%/60min drop on day0-1); Transit Swap (no token); Alpha Finance (no 5%/60min drop on day0-1); Cashio (error 400); Bitrue (no token); xToken (error 400); SwissBorg (error 400); EasyFi (no token); Wintermute (no token); Kronos Research (no 1min data (0 rows)); Elephant Money (error 400); Phemex (no token); Euler V1 (error 400); BtcTurk (no token); Poly Network (no token); CoinEx (error 400); Mixin Network (error 400); Ronin Bridge (no 1min data (0 rows)); CREAM Lending (error 400); Venus Protocol (no token); Ostium (no token); Step Finance (error 400); Bald (no token); IRA Financial / Gemini (no token); Nesa (no token); Pando Rings (no token); SBI Crypto (no token); Bitmart (error 400); Vee Finance (error 400); Upbit (no token); Nobitex (no token); Popsicle Finance (error 400); Tectonic (error 400); Nomad (no token); StableMagnet (no token); BXH (error 400); Deribit (no token); CoinDCX (no token); Sonne Finance (error 400); Infini (no token); Bunny (error 400)