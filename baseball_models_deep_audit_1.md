# Deep Audit: `seanpfarrell12-sys/baseball-models`
## Full Source Code Review — All 20 Pipeline Files Examined

**Date:** March 24, 2026  
**Scope:** Feature engineering, model architecture, alignment with framework document  
**Status:** All `01_input`, `02_build`, `03_analysis`, and `04_export` files reviewed

---

## Executive Summary

**This codebase is significantly more sophisticated than the initial audit suggested.** Having now read every pipeline file, the models implement nearly every advanced concept from the framework document — and in several cases go well beyond it. The moneyline model uses SIERA, xFIP, Statcast quality-of-contact, and pitch arsenal data. The totals model implements a physics-based air density engine with cosine-projected wind decomposition. The pitcher outs model uses a discrete-time survival framework with BF-level hazard expansion. The NRFI model isolates first-inning-specific Statcast pitch data and matches top-3 lineup slots against SP platoon splits.

**Revised overall grade: A- (up from B+)**

---

## Model-by-Model Deep Assessment

### 1. Moneyline Model — Grade: A-

#### What's Actually Implemented

**SP Feature Pipeline (confirmed in `01_input` + `02_build`):**
- SIERA and xFIP from FanGraphs — the doc's preferred ERA estimators ✅
- xwOBA against, barrel%, hard_hit%, whiff% from Statcast expected stats ✅
- Pitch arsenal: primary fastball velocity, spin rate, horizontal break, vertical break, fastball whiff%, offspeed weighted whiff% ✅
- SP handedness (binary L/R) ✅
- K%, BB%, K-BB% from FanGraphs ✅

**Bullpen Features:**
- Team bullpen pool ERA, K%, FIP computed from pitchers with GS < 5 ✅
- Differential features (home_bp_era - away_bp_era) ✅

**Lineup Features:**
- PA-weighted platoon wRC+, wOBA, K%, ISO vs opposing SP handedness ✅
- Correctly routes home lineup stats against away SP hand and vice versa ✅

**Context Features:**
- Static park factors (2024 estimates) ✅
- Home field indicator ✅
- 10 differential features computed (diff_sp_xwoba, diff_sp_siera, etc.) ✅

**Model Architecture (`03_analysis`):**
- XGBoost with isotonic probability calibration (CalibratedClassifierCV) ✅
- Walk-forward CV: Train<year → Test=year ✅
- Prior-year stats only (season-1 features for season N games) — no look-ahead ✅
- Regularization: L1 (0.5), L2 (2.0), min_child_weight=10 ✅
- Calibration holdout: 15% of most recent season ✅

**ID Pipeline:**
- Retrosheet SP IDs → Chadwick register → MLBAM + FanGraphs IDs ✅
- No name-based matching (explicitly noted as a fix from prior version) ✅

#### Remaining Gaps

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| No BaseRuns formula | Medium — would improve team-level run estimation vs. raw runs | Medium |
| No bullpen fatigue (PC L3/L5) | Medium — affects late-game moneyline edges | Medium |
| No umpire strike zone data | Low for moneyline specifically | Low |
| Static park factors (not dynamic) | Low — totals model has dynamic PF, could share | Low |

---

### 2. Totals (Over/Under) Model — Grade: A

This is the most impressive model in the repo.

#### What's Actually Implemented

**Physics-Based Dynamic Park Factor Engine (`02_build`):**
- Air density modeling: ρ ∝ (1/T) × (1 - 0.378 × Pv/P) ✅
- Temperature carry: +0.20% per °F above 65°F reference ✅
- Humidity carry: +0.04% per % RH above 50% (correctly notes humid air is LESS dense) ✅
- Wind outfield component via cosine projection: cos(angle between wind-toward vector and CF bearing) ✅
- Per-stadium CF bearing angles for accurate wind decomposition ✅
- Roof handling: fixed=0% weather effect, retractable=40%, open=100% ✅
- Altitude already absorbed in base park factor ✅
- City seasonal average temperature fallback when weather data is missing ✅

**Two-Equation System (`03_analysis`):**
- Negative Binomial NB2 (NOT Poisson!) — explicitly justified by overdispersion ✅
- Cameron & Trivedi overdispersion test confirming NB is necessary ✅
- Separate models for home_runs and away_runs ✅
- 200,000-sample Monte Carlo convolution for total runs distribution ✅
- Directly produces P(total > line) without normal approximation ✅

**Feature Set:**
- Home/away offense: wRC+, wOBA, OBP, SLG, ISO, K%, BB% ✅
- SP: SIERA, xFIP, K%, BB%, xwOBA against ✅
- Bullpen: ERA, K%, FIP ✅
- Weather: temperature, humidity, wind speed, wind direction ✅
- Dynamic PF decomposed into temp_factor, humidity_factor, wind_factor ✅
- Binary flags: is_cold_game (<50°F), is_hot_game (>85°F), is_coors, is_dome ✅
- Combined features: combined_off_wrc_plus, combined_sp_siera ✅
- Manager hook rates ✅

#### Remaining Gaps

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| No umpire strike zone data | Medium — doc calls this "significant yet overlooked" | Low |
| No DRS/UZR defensive metrics | Low-Medium — defensive quality affects run suppression | Medium |

---

### 3. NRFI/YRFI Model — Grade: A-

This model is far more sophisticated than typical NRFI approaches in the market.

#### What's Actually Implemented

**First-Inning-Specific Data (`01_input` + `02_build`):**
- Pulls raw Statcast pitch data filtered to inning=1 only ✅
- Computes SP first-inning stats from terminal pitch events per PA ✅
- First-inning K%, BB%, HR rate, contact quality — NOT season-level stats ✅
- Per-start then season-aggregate approach ✅

**YRFI Label Construction:**
- Built from Statcast score changes in first half-inning (top/bot) ✅
- Correctly handles both halves independently ✅

**Top-3 Lineup Features:**
- Retrosheet batting order slots 1-2-3 identified per game ✅
- Only hitters guaranteed to face SP in inning 1 ✅
- Platoon splits (wRC+, ISO, OBP, K%, BB%) matched to opposing SP handedness ✅
- Chadwick register bridges retrosheet IDs → MLBAM → FanGraphs IDs ✅

**Environmental Features:**
- Wind toward CF decomposition (cosine projection, same physics as totals) ✅
- Temperature HR carry factor (0.2% per °F above 70°F) ✅
- Altitude carry factor (1% per 1000 ft) ✅
- Park HR factor ✅
- Dome indicator (zeroes out weather effects) ✅
- Composite hr_environment index combining park + temp + altitude + wind ✅

**Interaction Features:**
- combined_fi_bb_pct (both SPs' first-inning walk rates) ✅
- home_lineup_vs_away_sp (top-3 wRC+ × (1 - SP K%)) ✅
- HR threat (hr_environment × top-3 ISO) ✅

**Model Architecture (`03_analysis`):**
- XGBoost binary classifier with isotonic regression calibration ✅
- Walk-forward temporal CV ✅
- Calibration curve / reliability diagram output ✅

#### Remaining Gaps

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| No first-pitch strike % for SPs | Medium — getting ahead in counts suppresses 1st-inning scoring | Low |
| No sprint speed for leadoff hitters | Low — affects scoring-from-first probability | Low |
| No SP pitch count from prior game | Low — fatigue could affect 1st-inning command | Medium |

---

### 4. Pitcher Outs Model — Grade: A

The most methodologically innovative model in the repo.

#### What's Actually Implemented

**Discrete-Time Survival Framework (`02_build`):**
- "Time" = batters faced (BF) 1..27 ✅
- "Event" = manager removes pitcher ✅
- Right-censoring for complete games (CG ≥ 24 outs) ✅
- BF-level dataset expansion: one row per (start × batter faced k) ✅

**TTOP Penalty — Fully Encoded:**
- bf_18_decision_point: binary flag at BF=18 (end of 2nd time through) ✅
- bf_19_ttop_start: binary flag at BF=19 (first batter of 3rd TTO) ✅
- is_ttop: binary for BF ≥ 19 ✅
- batters_into_ttop: max(0, bf_k - 18) — ramp function ✅
- times_through_order: ceil(k/9) ✅

**Pitch Count Accumulation:**
- est_pc_k = effective_ppp × bf_k (linear accumulation) ✅
- pc_fraction_k = est_pc_k / manager_typical_pc_limit ✅
- approaching_pc_limit: binary at 85% of limit ✅
- at_pc_limit: binary at 100% ✅
- past_hard_limit: binary at hard ceiling ✅
- pc_stress_k: ramp function max(0, pc_fraction - 0.7) × 10 ✅

**Manager Features:**
- Manager-specific typical PC limit and hard PC limit ✅
- Manager depth_score (willingness to go deep) ✅
- Interaction: ttop_x_low_patience (TTOP flag × (1 - depth_score)) ✅
- Raw manager removal stats + Bayesian priors ✅

**SP Skill Features:**
- K%, BB%, K-BB%, pitches_per_pa, CSW% ✅
- xwOBA against, barrel%, hard_hit% ✅
- obp_proxy for estimating P(out per PA) ✅

**Opponent Features:**
- Opponent batting OBP (affects scoring rate and PC accumulation) ✅

**Monte Carlo Simulation (`03_analysis`):**
- Simulates games: at each BF, draws Bernoulli(h_k) for removal ✅
- Draws Bernoulli(p_out) for out on each PA ✅
- Forced removal at hard PC limit ✅
- Full empirical distribution over outs (no normal assumption) ✅
- Directly reads P(outs > line) from simulation histogram ✅

**Cox PH Companion:**
- Optional lifelines CoxPH for hazard ratio interpretation ✅

#### Remaining Gaps

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| No bullpen fatigue tracking (PC L3/L5 for relievers) | Medium — affects manager's willingness to push SP | Medium |
| No in-game score differential effect | Low — losing managers may push SP longer | Medium |

---

### 5. Hitter Total Bases Model — Grade: A-

#### What's Actually Implemented

**Batter Features (prior-year Statcast):**
- xBA, xSLG, xwOBA (expected stats from Statcast) ✅
- Exit velocity average ✅
- Barrel batted rate ✅
- Launch angle average ✅
- Hard hit percent ✅

**Platoon Features:**
- wRC+, wOBA, K%, ISO vs opposing SP handedness ✅
- FanGraphs splits API with proper platoon code routing ✅

**SP Features:**
- SP handedness ✅
- SP xwOBA against, barrel%, hard_hit% ✅
- Full arsenal: FB velo, spin, H-break, V-break, FB whiff%, FB usage ✅
- Offspeed whiff%, offspeed usage ✅

**Interaction Features:**
- ev_vs_sp_xwoba: batter EV × SP xwOBA against ✅
- barrel_vs_sp_barrel: batter barrel% - SP barrel% against ✅

**Model Architecture (`03_analysis`):**
- XGBoost multi:softprob (5-class multinomial) — NOT binary ✅
- Full probability distribution: P(TB=0), P(TB=1), P(TB=2), P(TB=3), P(TB≥4) ✅
- PA volume weighting via Monte Carlo Poisson convolution ✅
- Batting slot → PA projection (slot 1=4.33 PA, slot 9=3.90 PA) ✅
- Class imbalance correction via sample weights ✅
- Walk-forward temporal CV ✅

**ID Pipeline:**
- MLBAM numeric keys throughout (not name-matching) ✅
- Chadwick register bridges retrosheet → MLBAM → FanGraphs ✅
- Retrosheet game logs for confirmed batting order slots ✅

#### Remaining Gaps

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| No launch angle standard deviation (SD LA) | Medium — doc specifically highlights this as a consistency signal | Low |
| No park factor in hitter TB | Medium — Coors Field batter should get a boost | Low |
| No recent form / rolling window features | Low — last 7/14 day splits could capture hot/cold streaks | Medium |

---

## Cross-Model Architecture Strengths

These are things done well across all 5 models:

1. **Prior-year features only** — Every model uses (season - 1) stats to avoid look-ahead bias. This is the single most common mistake in sports ML and it's correctly avoided everywhere.

2. **MLBAM numeric ID joins** — All joins use the Chadwick Bureau register to bridge retrosheet → MLBAM → FanGraphs IDs. No name-string matching anywhere. The hitter TB doc even notes this as a fix for a prior "0 hitters matched" bug.

3. **Walk-forward temporal CV** — Every model uses train-on-prior-seasons, test-on-current-season cross-validation. No random shuffled splits that would leak future information.

4. **Probability calibration** — Moneyline and NRFI use isotonic regression calibration. The totals model uses NB distributional calibration. The pitcher outs model uses Monte Carlo simulation. The hitter TB model uses multinomial softprob. None of the models output raw uncalibrated scores.

5. **Full distributions, not point estimates** — Totals: Monte Carlo NB convolution. Pitcher outs: MC survival simulation. Hitter TB: 5-class probability vector with PA convolution. This is vastly superior to point-estimate approaches.

6. **Physics-grounded environmental modeling** — The totals and NRFI models share a physics-based air density framework with cosine-projected wind decomposition, temperature carry factors, and altitude corrections. This is research-grade work.

---

## Remaining Improvement Opportunities (Prioritized)

### HIGH PRIORITY

**1. Umpire Strike Zone Data**
Affects: Totals, NRFI
The doc calls the home plate umpire "significant yet overlooked." Umpires with larger zones produce more strikeouts and fewer walks, suppressing scoring. Historical umpire BB/K ratios and O/U run tendencies are available from UmpScorecards.com and could be added as features to both the totals and NRFI models with relatively low effort.

**2. Bullpen Fatigue Tracking (PC L3/L5)**
Affects: Moneyline, Pitcher Outs
Neither model currently tracks reliever workload over the prior 3-5 days. For the moneyline model, a gassed bullpen makes the opposing team more likely to win in close late-inning games. For the pitcher outs model, bullpen fatigue directly affects the manager's decision to push the starter deeper. Sources like Outlier track this data daily.

### MEDIUM PRIORITY

**3. Launch Angle Standard Deviation**
Affects: Hitter TB
The doc specifically highlights SD LA as underused — a batter with a "perfect" 15° average LA but high variance is actually producing lots of pop-ups and grounders. Computing rolling SD LA from pitch-level Statcast data would improve the TB model's prediction of contact consistency.

**4. Park Factor for Hitter TB**
Affects: Hitter TB
The hitter TB model currently has no park factor at all. A batter facing an SP at Coors Field should have a meaningfully higher TB expectation than the same matchup at Oracle Park. The totals model already has dynamic park factors that could be shared.

**5. BaseRuns for Moneyline**
Affects: Moneyline
The moneyline model uses team-level platoon wRC+ but doesn't compute BaseRuns (the doc's A·B/(B+C)+D formula). BaseRuns would better isolate true team offensive talent from sequencing luck. This is a medium-difficulty addition.

**6. Defensive Metrics (DRS/UZR)**
Affects: Totals
The doc notes that elite defenses "effectively lower the game's total by converting more batted balls into outs." Adding team DRS or UZR as features to the totals model would capture defensive run prevention that isn't reflected in pitching stats alone.

### LOW PRIORITY

**7. First-Pitch Strike % for NRFI SPs**
The first pitch of each PA in inning 1 sets the count direction. SPs who throw first-pitch strikes at high rates are more efficient in the first inning. This is a simple feature to derive from the existing Statcast first-inning data.

**8. Sprint Speed for NRFI Leadoff Hitters**
Fast leadoff hitters who reach base are more likely to score on a subsequent single. Statcast sprint speed data is available and could be added to the top-3 lineup features.

**9. Cross-Model Feature Sharing**
The totals model's dynamic park factor engine could feed into the hitter TB and NRFI models. The pitcher outs model's manager hazard data could inform the moneyline model's bullpen assessment. Currently the 5 models run independently.

**10. Backtesting Framework**
`grade_daily.py` grades forward performance, but there's no retroactive backtesting module. A `backtest.py` replaying models against 2-3 prior seasons would help validate improvements before deployment.

---

## Revised Scorecard

| Category | Initial Grade | Revised Grade | Key Upgrade Reason |
|----------|:---:|:---:|-----|
| Architecture & Code Quality | A | A | Confirmed: clean separation, fault isolation |
| Data Pipeline | A- | A | MLBAM ID joins throughout, Chadwick register |
| Moneyline Model | B+ | **A-** | SIERA + xFIP + arsenal + platoon splits confirmed |
| Totals Model | A- | **A** | Physics-based dynamic PF + NB2 + Monte Carlo |
| Hitter TB Model | B | **A-** | Multinomial softprob + MC PA convolution + arsenal matchups |
| Pitcher Outs Model | B- | **A** | Discrete-time survival + TTOP + manager hazard + MC sim |
| NRFI/YRFI Model | B | **A-** | First-inning Statcast + top-3 platoon + environmental physics |
| Grading & Tracking | A | A | Confirmed |
| Bet Sizing & Calibration | C+ | **B+** | Isotonic calibration + distributional outputs confirmed |

**Overall: A- (up from B+)**

The initial audit underestimated this codebase significantly because the feature engineering and model architecture were hidden in the subdirectory files. With full source access, it's clear this is a research-grade predictive modeling system that implements virtually every recommendation from the framework document and adds several innovations beyond it (survival framing for pitcher outs, NB2 with MC convolution for totals, multinomial TB with PA convolution).

The highest-impact improvements remaining are umpire data for totals/NRFI and bullpen fatigue tracking for moneyline/pitcher outs.
