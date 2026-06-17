#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, random, urllib.request, urllib.error, datetime, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "portfolio.json")
OUT = os.path.join(ROOT, "data", "output.json")
UA = {"User-Agent": "Mozilla/5.0 (compatible; CarteraQuant/1.0)"}
WARN = []

def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def yahoo_history(symbol, rng="2y"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval=1d"
    try:
        d = _get(url)
        res = d["chart"]["result"][0]
        ts = res["timestamp"]
        closes = res["indicators"]["quote"][0]["close"]
        cur = res["meta"].get("currency", "EUR")
        out = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            day = datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
            out.append((day, float(c)))
        m = {}
        for day, c in out:
            m[day] = c
        return sorted(m.items()), cur
    except Exception as e:
        WARN.append(f"No se pudo descargar {symbol}: {e}")
        return None, None

def yahoo_fundamentals(symbol):
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=summaryDetail,defaultKeyStatistics"
    try:
        d = _get(url)
        r = d["quoteSummary"]["result"][0]
        sd = r.get("summaryDetail", {})
        ks = r.get("defaultKeyStatistics", {})
        pe = sd.get("trailingPE", {}).get("raw") or ks.get("trailingPE", {}).get("raw")
        dy = sd.get("dividendYield", {}).get("raw")
        return (round(pe, 1) if pe else None, round(dy * 100, 2) if dy else None)
    except Exception:
        return (None, None)

def fx_to_eur_series(cur):
    if cur == "EUR":
        return None
    series, _ = yahoo_history(f"{cur}EUR=X", "2y")
    if not series:
        WARN.append(f"No se pudo obtener tipo de cambio {cur}->EUR; se asume 1.")
        return {}
    return dict(series)

def returns(series):
    rs = []
    for i in range(1, len(series)):
        p0 = series[i - 1][1]
        if p0 > 0:
            rs.append((series[i][0], series[i][1] / p0 - 1))
    return rs

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    return sxy / math.sqrt(sx * sy)

def xirr(cashflows):
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    def npv(r):
        return sum(cf / (1 + r) ** ((d - t0).days / 365.0) for d, cf in cashflows)
    lo, hi = -0.9999, 20.0
    flo, fhi = npv(lo), npv(hi)
    if flo * fhi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if abs(fm) < 1e-7:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2

def portfolio_series(rets_by_id, weights):
    ids = [i for i in rets_by_id if rets_by_id[i]]
    if not ids:
        return []
    maps = {i: dict(rets_by_id[i]) for i in ids}
    common = set(maps[ids[0]].keys())
    for i in ids[1:]:
        common &= set(maps[i].keys())
    common = sorted(common)
    out = []
    for day in common:
        r = sum(weights.get(i, 0) * maps[i][day] for i in ids)
        out.append((day, r))
    return out

def risk_metrics(series_rets, rf):
    rs = [r for _, r in series_rets]
    n = len(rs)
    if n < 8:
        return {"ok": False, "n": n}
    mean = sum(rs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rs) / (n - 1))
    ann_vol = sd * math.sqrt(252) * 100
    geo = 1.0
    for x in rs:
        geo *= (1 + x)
    ann_ret = (geo ** (252 / n) - 1) * 100
    cum = (geo - 1) * 100
    dn = math.sqrt(sum(min(x, 0) ** 2 for x in rs) / n) * math.sqrt(252) * 100
    peak = c = 1.0
    mdd = 0.0
    for x in rs:
        c *= (1 + x)
        peak = max(peak, c)
        mdd = min(mdd, c / peak - 1)
    pos = sum(1 for x in rs if x > 0) / n * 100
    var95 = (mean - 1.645 * sd) * 100
    return {
        "ok": True, "n": n,
        "twr": round(cum, 2), "ann_ret": round(ann_ret, 2), "ann_vol": round(ann_vol, 1),
        "sharpe": round((ann_ret - rf) / ann_vol, 2) if ann_vol else 0,
        "sortino": round((ann_ret - rf) / dn, 2) if dn else 0,
        "max_dd": round(mdd * 100, 1),
        "calmar": round(ann_ret / abs(mdd * 100), 2) if mdd < 0 else 0,
        "pos_days": round(pos, 0), "var95": round(var95, 1),
    }

def cov_matrix(rets_by_id, ids):
    maps = {i: dict(rets_by_id[i]) for i in ids}
    common = None
    for i in ids:
        s = set(maps[i].keys())
        common = s if common is None else (common & s)
    common = sorted(common or [])
    series = {i: [maps[i][d] for d in common] for i in ids}
    n = len(common)
    mu = {i: (sum(series[i]) / n if n else 0) for i in ids}
    cov = {}
    for a in ids:
        cov[a] = {}
        for b in ids:
            if n < 2:
                cov[a][b] = 0
            else:
                cov[a][b] = sum((series[a][k] - mu[a]) * (series[b][k] - mu[b]) for k in range(n)) / (n - 1) * 252
    mu_ann = {i: mu[i] * 252 for i in ids}
    corr = {}
    for a in ids:
        corr[a] = {}
        for b in ids:
            va, vb = cov[a][a], cov[b][b]
            corr[a][b] = round(cov[a][b] / math.sqrt(va * vb), 2) if va > 0 and vb > 0 else 0
    return mu_ann, cov, corr, n

def risk_contributions(ids, weights, cov):
    w = [weights.get(i, 0) for i in ids]
    mc, total = [], 0.0
    for i, a in enumerate(ids):
        c = sum(w[j] * cov[a][ids[j]] for j in range(len(ids)))
        mc.append(w[i] * c)
        total += w[i] * c
    return {ids[i]: (round(mc[i] / total * 100, 1) if total > 0 else 0) for i in range(len(ids))}

def efficient_frontier(ids, mu_ann, cov, cap, rf, nsims=6000):
    if len(ids) < 2:
        return None
    def port(w):
        ret = sum(w[i] * mu_ann[ids[i]] for i in range(len(ids))) * 100
        var = 0.0
        for i in range(len(ids)):
            for j in range(len(ids)):
                var += w[i] * w[j] * cov[ids[i]][ids[j]]
        vol = math.sqrt(max(var, 0)) * 100
        return ret, vol
    cap = max(cap, 1.0 / len(ids) + 1e-9)
    samples = []
    tries = 0
    while len(samples) < nsims and tries < nsims * 8:
        tries += 1
        raw = [-math.log(random.random()) for _ in ids]
        s = sum(raw)
        w = [x / s for x in raw]
        if max(w) > cap:
            continue
        ret, vol = port(w)
        samples.append({"w": w, "ret": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol > 0 else -99})
    if not samples:
        return None
    maxs = max(samples, key=lambda p: p["sharpe"])
    minv = min(samples, key=lambda p: p["vol"])
    bins = {}
    for p in samples:
        k = round(p["vol"] / 1.5)
        if k not in bins or p["ret"] > bins[k]["ret"]:
            bins[k] = p
    line = sorted(bins.values(), key=lambda p: p["vol"])
    def pack(p):
        return {"vol": round(p["vol"], 2), "ret": round(p["ret"], 2), "sharpe": round(p["sharpe"], 2),
                "weights": {ids[i]: round(p["w"][i] * 100, 1) for i in range(len(ids)) if p["w"][i] > 0.01}}
    cloud = [{"vol": round(p["vol"], 2), "ret": round(p["ret"], 2)} for p in random.sample(samples, min(1200, len(samples)))]
    return {"cloud": cloud, "line": [{"vol": round(p["vol"], 2), "ret": round(p["ret"], 2)} for p in line],
            "max_sharpe": pack(maxs), "min_var": pack(minv)}

def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    st = cfg.get("settings", {})
    rf = st.get("risk_free_pct", 2.5)
    cap = st.get("max_weight_pct", 60) / 100.0
    tol = st.get("rebalance_tolerance_pts", 5)
    rng = st.get("lookback", "2y")
    assets = cfg["assets"]
    txns = cfg["transactions"]
    prices_eur = {}
    last_price = {}
    fundamentals = {}
    for a in assets:
        series, cur = yahoo_history(a["symbol"], rng)
        if not series:
            buys = [t for t in txns if t["asset"] == a["id"]]
            if buys:
                last_price[a["id"]] = buys[-1]["price"]
                WARN.append(f"{a['name']}: sin datos de mercado; uso ultimo precio de compra ({buys[-1]['price']}).")
            continue
        if cur and cur != "EUR":
            fx = fx_to_eur_series(cur)
            if fx:
                conv, lastfx = [], None
                for day, p in series:
                    if day in fx:
                        lastfx = fx[day]
                    if lastfx:
                        conv.append((day, p * lastfx))
                series = conv or series
        prices_eur[a["id"]] = series
        last_price[a["id"]] = series[-1][1]
        ytd = None
        yr = datetime.date.today().year
        first = next((p for d, p in series if d >= f"{yr}-01-01"), None)
        if first and series[-1][1]:
            ytd = round((series[-1][1] / first - 1) * 100, 1)
        pe, dy = (None, None)
        if a.get("cls") != "crypto":
            pe, dy = yahoo_fundamentals(a["symbol"])
        fundamentals[a["id"]] = {"pe": pe, "ytd": ytd, "dy": dy}
    holdings = []
    total_val = total_inv = 0.0
    for a in assets:
        units = sum((t["amount"] - t.get("fee", 0)) / t["price"] for t in txns if t["asset"] == a["id"] and t["price"] > 0)
        inv = sum(t["amount"] for t in txns if t["asset"] == a["id"])
        price = last_price.get(a["id"], 0)
        val = units * price
        total_val += val
        total_inv += inv
        holdings.append({"id": a["id"], "name": a["name"], "cls": a.get("cls"),
                         "target": a.get("target", 0), "units": round(units, 8),
                         "invested": round(inv, 2), "price": round(price, 4),
                         "value": round(val, 2), "pnl": round(val - inv, 2),
                         "pnl_pct": round((val - inv) / inv * 100, 1) if inv else 0})
    for h in holdings:
        h["weight"] = round(h["value"] / total_val * 100, 1) if total_val else 0
        h["drift"] = round(h["weight"] - h["target"], 1)
    weights = {h["id"]: (h["value"] / total_val if total_val else 0) for h in holdings}
    rets_by_id = {i: returns(prices_eur[i]) for i in prices_eur}
    ids = [i for i in rets_by_id if len(rets_by_id[i]) >= 8]
    port = portfolio_series(rets_by_id, weights)
    metrics = risk_metrics(port, rf)
    correlations = {"ids": [], "matrix": []}
    frontier = None
    if len(ids) >= 1:
        mu_ann, cov, corr, ncommon = cov_matrix(rets_by_id, ids)
        correlations = {"ids": ids, "matrix": [[corr[a][b] for b in ids] for a in ids], "n": ncommon}
        rc = risk_contributions(ids, weights, cov)
        for h in holdings:
            if h["id"] in ids:
                h["ann_vol"] = round(math.sqrt(cov[h["id"]][h["id"]]) * 100, 1)
                h["risk_contrib"] = rc.get(h["id"], 0)
        if len(ids) >= 2:
            frontier = efficient_frontier(ids, mu_ann, cov, cap, rf)
            def pt(wmap):
                ret = sum(wmap.get(i, 0) * mu_ann[i] for i in ids) * 100
                var = sum(wmap.get(i, 0) * wmap.get(j, 0) * cov[i][j] for i in ids for j in ids)
                return {"vol": round(math.sqrt(max(var, 0)) * 100, 2), "ret": round(ret, 2)}
            tt = sum(a.get("target", 0) for a in assets) or 1
            twmap = {a["id"]: a.get("target", 0) / tt for a in assets if a["id"] in ids}
            frontier["current"] = pt(weights)
            frontier["target"] = pt(twmap)
            frontier["assets"] = [{"id": i, "vol": round(math.sqrt(cov[i][i]) * 100, 1), "ret": round(mu_ann[i] * 100, 1)} for i in ids]
    idx, v = [], 100.0
    for day, r in port:
        v *= (1 + r)
        idx.append({"date": day, "index": round(v, 2)})
    cfs = [(datetime.date.fromisoformat(t["date"]), -t["amount"]) for t in txns]
    cfs.sort()
    if total_val > 0:
        cfs.append((datetime.date.today(), total_val))
    tir = xirr(cfs)
    opportunities = []
    if frontier and ids:
        mu_ann2, cov2, _, _ = cov_matrix(rets_by_id, ids)
        base = [a for a in assets if a.get("target", 0) > 0 and a["id"] in ids]
        for a in assets:
            f = fundamentals.get(a["id"], {})
            pe = f.get("pe")
            bucket = "s/d" if pe is None else ("barato" if pe < 15 else "razonable" if pe < 25 else "exigente" if pe < 35 else "caro")
            avg_corr = None
            if a["id"] in ids and base:
                tt = sum(b.get("target", 0) for b in base) or 1
                avg_corr = 0.0
                for b in base:
                    va, vb = cov2[a["id"]][a["id"]], cov2[b["id"]][b["id"]]
                    c = cov2[a["id"]][b["id"]] / math.sqrt(va * vb) if va > 0 and vb > 0 else 0
                    avg_corr += (b.get("target", 0) / tt) * c
                avg_corr = round(avg_corr, 2)
            opportunities.append({"id": a["id"], "name": a["name"], "pe": pe, "ytd": f.get("ytd"), "bucket": bucket, "avg_corr": avg_corr})
    out = {
        "updated": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "currency": "EUR",
        "settings": {"risk_free_pct": rf, "rebalance_tolerance_pts": tol, "max_weight_pct": st.get("max_weight_pct", 60)},
        "totals": {"value": round(total_val, 2), "invested": round(total_inv, 2),
                   "pnl": round(total_val - total_inv, 2),
                   "pnl_pct": round((total_val - total_inv) / total_inv * 100, 1) if total_inv else 0,
                   "xirr": round(tir * 100, 1) if tir is not None else None},
        "metrics": metrics, "holdings": holdings, "index": idx,
        "correlations": correlations, "frontier": frontier,
        "opportunities": opportunities, "warnings": WARN,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"OK. Valor {out['totals']['value']} EUR. Avisos: {len(WARN)}")

if __name__ == "__main__":
    main()
