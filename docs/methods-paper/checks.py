import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(20261104)

print("A. Spurious timing correlation: corr(level X_t, X_{t+h}-X_t) in a sample of length T")
T = 1000
for h in (5, 20, 60):
    L = T - h
    rho = []
    for _ in range(4000):
        x = np.cumsum(rng.normal(size=T))
        rho.append(np.corrcoef(x[:L], x[h:h + L] - x[:L])[0, 1])
    print(f"  h={h:2d}: simulated mean {np.mean(rho):+.3f} | formula -sqrt(3h/2L) = {-np.sqrt(1.5 * h / L):+.3f}")

print("B. Share of next year's values outside the training range (3 training years, 1 year after)")
n_tr, n_te = 756, 252
for mu_ann, sig_ann, label in [(0.0, 0.25, "random walk"), (0.10, 0.25, "drift 10%, vol 25%"), (0.20, 0.25, "drift 20%, vol 25%")]:
    out = []
    for _ in range(4000):
        x = np.cumsum(rng.normal(mu_ann / 252, sig_ann / np.sqrt(252), n_tr + n_te))
        lo, hi = x[:n_tr].min(), x[:n_tr].max()
        out.append(((x[n_tr:] < lo) | (x[n_tr:] > hi)).mean())
    print(f"  {label:20s}: mean share outside {np.mean(out):.2f}")

print("C. Overlapping trades: per-trade Sharpe of k consecutive-day entries, no skill (zero drift)")
d, r, sig = 30, 0.05, 0.02
for k in (3, 5, 10, 20):
    srs = []
    for _ in range(4000):
        p = 100 * np.exp(np.cumsum(rng.normal(0, sig, k + d)))
        rets = []
        for i in range(k):
            w = p[i:i + d]
            hit = np.nonzero((w - w[0]) / w[0] > r)[0]
            sell = w[hit[0]] if len(hit) else w[-1]
            rets.append(sell / w[0] - 1 - 0.00025)
        rets = np.array(rets)
        s = rets.std(ddof=1)
        if s > 0:
            srs.append((rets.mean() - 0.00022) / s)
    srs = np.array(srs)
    print(f"  k={k:2d}: median |SR| {np.median(np.abs(srs)):.2f}, share |SR|>2: {(np.abs(srs) > 2).mean():.2f}, share SR>1: {(srs > 1).mean():.2f}")
# independent trades for comparison
srs = []
for _ in range(4000):
    rets = []
    for i in range(5):
        p = 100 * np.exp(np.cumsum(rng.normal(0, sig, d)))
        hit = np.nonzero((p - p[0]) / p[0] > r)[0]
        sell = p[hit[0]] if len(hit) else p[-1]
        rets.append(sell / p[0] - 1 - 0.00025)
    rets = np.array(rets); srs.append((rets.mean() - 0.00022) / rets.std(ddof=1))
print(f"  5 independent trades: median |SR| {np.median(np.abs(srs)):.2f}")
# correlation of two windows offset by one day (no take-profit): (d-1)/d
print(f"  window overlap correlation (d-1)/d = {(d - 1) / d:.3f}; inflation 1/sqrt(1-rho) = {np.sqrt(d):.2f}")

print("D. Selection from noise: expected best training Sharpe among N independent no-skill rules")
for years in (3, 4):
    se = np.sqrt(1 / years)   # annualised Sharpe standard error, zero true Sharpe, daily returns
    for N in (100, 1000, 10000):
        # exact expectation of the max of N standard normals
        z = rng.normal(size=(2000, N)).max(axis=1).mean()
        print(f"  {years} training years, N={N:5d}: E[max SR] = {se * z:.2f} (approx sqrt(2 ln N)/sqrt(years) = {np.sqrt(2*np.log(N)/years):.2f})")

print("E. Fundamental law: IC needed for an information ratio of 0.5 before costs")
for label, breadth in [("one stock, 20-day bets", 252 / 20), ("one stock, 5-day bets", 252 / 5), ("336 stocks, monthly, indep.", 336 * 12), ("336 stocks, monthly, 30 eff. bets", 30 * 12)]:
    print(f"  {label:32s}: breadth {breadth:7.0f}/yr, IC needed {0.5 / np.sqrt(breadth):.3f}")

print("F. Standard error of an annualised Sharpe ratio (true SR 0.5, normal returns)")
for years in (1, 4, 7):
    print(f"  {years} year(s): {np.sqrt((1 + 0.5**2 / 2) / years):.2f}")

print("G. Random-date control: 7% of 179 runs above the 95th percentile under a uniform null")
from scipy.stats import binomtest
print(f"  binomial p (one-sided) = {binomtest(round(0.07*179), 179, 0.05, alternative='greater').pvalue:.2f}")
