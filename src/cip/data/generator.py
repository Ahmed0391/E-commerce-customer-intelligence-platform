"""Latent-state simulator for a synthetic e-commerce world.

Design principle
----------------
Every customer has a *hidden* state (purchase rate, dropout propensity, category
affinities, price preference, exposure to delivery problems). All observable data
(orders, browsing events, support conversations) is sampled FROM that state. Nothing
observable is generated from another observable by a hand-written rule.

This matters for evaluation: the pipeline only sees observables and has to infer the
hidden state (e.g. "is this customer still active?"). Because we know the truth, we
can measure how well it does, and the error is never zero, just like in reality.

Purchase process (BG/NBD-style, Fader, Hardie & Lee 2005)
---------------------------------------------------------
- While active, purchases arrive as a Poisson process with rate lambda_i.
- After each purchase the customer drops out with probability p_i
  (+ a boost if that order had a delivery problem -> the churn/complaint link).
- lambda_i ~ Gamma, p_i ~ Beta across customers: heterogeneity is the whole point.

Outputs
-------
data/raw/          observable tables  (the pipeline may read these)
data/ground_truth/ latent state, conversation labels, oracle actions
                   (ONLY evaluation code may read these)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cip.data.catalog import make_campaigns, make_catalog
from cip.data.oracle import best_actions

ACQ_CHANNELS = ["organic", "paid_search", "social", "referral"]  # deliberately non-informative
DAY = pd.Timedelta(days=1)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _choice(rng: np.random.Generator, dist: dict[str, float]) -> str:
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


@dataclass
class _Tables:
    """Row buffers. Appending dicts to lists is simple and fast enough for ~10^5 rows."""
    orders: list = field(default_factory=list)
    items: list = field(default_factory=list)
    events: list = field(default_factory=list)
    convs: list = field(default_factory=list)
    conv_labels: list = field(default_factory=list)
    next_order: int = 1
    next_session: int = 1
    next_conv: int = 1


class Simulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        self.snapshot = pd.Timestamp(cfg["snapshot_date"])
        self.products = make_catalog(self.rng)
        self.campaigns = make_campaigns(self.snapshot)
        self.categories = list(self.products["category"].unique())
        # Per-category product arrays, for fast sampling
        self._cat_products = {
            c: (g["product_id"].to_numpy(), g["price"].to_numpy())
            for c, g in self.products.groupby("category", sort=False)
        }
        self.t = _Tables()

    # ------------------------------------------------------------------ latent
    def sample_latent(self) -> pd.DataFrame:
        cfg, rng, n = self.cfg["latent"], self.rng, self.cfg["n_customers"]
        start, end = pd.Timestamp(self.cfg["signup_start"]), pd.Timestamp(self.cfg["signup_end"])
        span_days = (end - start).days
        affinity = rng.dirichlet(np.full(len(self.categories), cfg["category_affinity_alpha"]), size=n)
        df = pd.DataFrame({
            "customer_id": np.arange(1, n + 1),
            "signup_date": start + pd.to_timedelta(rng.integers(0, span_days + 1, n), unit="D"),
            "acquisition_channel": rng.choice(ACQ_CHANNELS, size=n),
            "purchase_rate": rng.gamma(cfg["purchase_rate"]["shape"], cfg["purchase_rate"]["scale"], n),
            "dropout_prob": rng.beta(cfg["dropout_prob"]["a"], cfg["dropout_prob"]["b"], n),
            "price_preference": rng.normal(cfg["price_preference"]["mean"], cfg["price_preference"]["sd"], n),
            "issue_rate": rng.beta(cfg["issue_rate"]["a"], cfg["issue_rate"]["b"], n),
        })
        for j, c in enumerate(self.categories):
            df[f"affinity_{c}"] = affinity[:, j]
        df["top_category"] = [self.categories[j] for j in affinity.argmax(axis=1)]
        return df

    # ------------------------------------------------------------ helpers
    def _product_weights(self, cat: str, tau: float) -> np.ndarray:
        _, prices = self._cat_products[cat]
        w = prices ** tau
        return w / w.sum()

    def _sample_products(self, affinity: np.ndarray, tau: float, k: int) -> list[tuple[int, float, str]]:
        out = []
        for _ in range(k):
            cat = self.categories[self.rng.choice(len(self.categories), p=affinity)]
            ids, prices = self._cat_products[cat]
            j = self.rng.choice(len(ids), p=self._product_weights(cat, tau))
            out.append((int(ids[j]), float(prices[j]), cat))
        return out

    def _event(self, cid, sid, ts, etype, pid=None, oid=None):
        self.t.events.append({"customer_id": cid, "session_id": sid, "ts": ts,
                              "event_type": etype, "product_id": pid, "order_id": oid})

    def _conversation(self, cid, ts, intent, trigger, frustration, order_id=None):
        if ts > self.snapshot:
            return
        cid_ = self.t.next_conv
        self.t.next_conv += 1
        self.t.convs.append({"conversation_id": cid_, "customer_id": cid, "ts": ts,
                             "order_id": order_id, "channel": "chat", "text": None})
        self.t.conv_labels.append({"conversation_id": cid_, "intent": intent, "trigger": trigger,
                                   "frustration": round(float(frustration), 3)})

    # ------------------------------------------------------------- sessions
    def _purchase_session(self, cid, ts, affinity, tau, issue_rate):
        """view -> add_to_cart -> checkout -> purchase. Returns (order_id, had_issue)."""
        rng, ocfg = self.rng, self.cfg["orders"]
        sid = self.t.next_session
        self.t.next_session += 1
        oid = self.t.next_order
        self.t.next_order += 1
        k = int(rng.integers(ocfg["items_per_order"]["min"], ocfg["items_per_order"]["max"] + 1))
        items = self._sample_products(affinity, tau, k)
        extra_views = self._sample_products(affinity, tau, int(rng.integers(0, 3)))
        t = ts - pd.Timedelta(minutes=int(rng.integers(5, 40)))
        for pid, _, _ in extra_views + items:
            self._event(cid, sid, t, "view", pid)
            t += pd.Timedelta(minutes=int(rng.integers(1, 5)))
        for pid, _, _ in items:
            self._event(cid, sid, t, "add_to_cart", pid)
            t += pd.Timedelta(minutes=1)
        self._event(cid, sid, t, "checkout")
        self._event(cid, sid, ts, "purchase", oid=oid)

        had_issue = rng.random() < issue_rate
        status = _choice(rng, ocfg["issue_types"]) if had_issue else "on_time"
        delivery = ts + DAY * int(rng.integers(ocfg["delivery_days"]["min"], ocfg["delivery_days"]["max"] + 1))
        if status == "late_delivery":
            delivery += DAY * int(rng.integers(3, 10))
        value = round(sum(p for _, p, _ in items), 2)
        self.t.orders.append({"order_id": oid, "customer_id": cid, "order_ts": ts, "n_items": k,
                              "order_value": value, "delivery_date": delivery, "delivery_status": status})
        for pid, price, _ in items:
            self.t.items.append({"order_id": oid, "product_id": pid, "unit_price": price})
        return oid, had_issue, delivery

    def _browse_session(self, cid, ts, affinity, tau):
        """A session that does NOT end in a purchase: views, maybe a cart, maybe an abandoned checkout."""
        rng, ecfg = self.rng, self.cfg["events"]
        sid = self.t.next_session
        self.t.next_session += 1
        n_views = int(rng.integers(ecfg["views_per_session"]["min"], ecfg["views_per_session"]["max"] + 1))
        viewed = self._sample_products(affinity, tau, n_views)
        t = ts
        for pid, _, _ in viewed:
            self._event(cid, sid, t, "view", pid)
            t += pd.Timedelta(minutes=int(rng.integers(1, 6)))
        if rng.random() < ecfg["p_add_to_cart_browse"]:
            self._event(cid, sid, t, "add_to_cart", viewed[0][0])
            if rng.random() < ecfg["p_checkout_given_cart"]:
                self._event(cid, sid, t + pd.Timedelta(minutes=2), "checkout")

    def _poisson_times(self, start: pd.Timestamp, end: pd.Timestamp, rate_per_day: float) -> list[pd.Timestamp]:
        days = (end - start) / DAY
        if days <= 0 or rate_per_day <= 0:
            return []
        n = self.rng.poisson(rate_per_day * days)
        return sorted(start + DAY * self.rng.uniform(0, days, n))

    # ------------------------------------------------------------ customer
    def simulate_customer(self, row) -> dict:
        rng, cfg = self.rng, self.cfg
        cid = int(row.customer_id)
        affinity = np.array([row[f"affinity_{c}"] for c in self.categories])
        tau, lam = float(row.price_preference), float(row.purchase_rate)
        t, alive, dropout_ts = row.signup_date, True, pd.NaT

        # 1) Purchases: Poisson arrivals, BG/NBD dropout after each purchase
        while True:
            t = t + DAY * rng.exponential(1.0 / lam)
            if t > self.snapshot:
                break
            oid, had_issue, delivery = self._purchase_session(cid, t, affinity, tau, row.issue_rate)
            if had_issue and rng.random() < cfg["conversations"]["p_contact_given_issue"]:
                self._conversation(cid, delivery + DAY * rng.uniform(0, 3),
                                   _choice(rng, cfg["conversations"]["issue_intents"]),
                                   "delivery_issue", rng.beta(5, 2), order_id=oid)
            p = min(1.0, row.dropout_prob + (cfg["latent"]["issue_dropout_boost"] if had_issue else 0.0))
            if rng.random() < p:
                alive, dropout_ts = False, t
                if rng.random() < cfg["conversations"]["p_exit_signal_at_dropout"]:
                    self._conversation(cid, t + DAY * rng.uniform(5, 30), "delete_account",
                                       "exit", rng.beta(3, 3))
                break

        active_end = dropout_ts if not alive else self.snapshot

        # 2) Browsing sessions (no purchase): proportional to purchase rate while active,
        #    a trickle after dropout.
        b = cfg["events"]["browse_to_purchase_ratio"] * lam
        for ts in self._poisson_times(row.signup_date, active_end, b):
            self._browse_session(cid, ts, affinity, tau)
        if not alive:
            for ts in self._poisson_times(dropout_ts, self.snapshot, b * cfg["events"]["post_dropout_browse_factor"]):
                self._browse_session(cid, ts, affinity, tau)

        # 3) Routine support contacts while active
        for ts in self._poisson_times(row.signup_date, active_end,
                                      cfg["conversations"]["background_rate_per_year"] / 365):
            self._conversation(cid, ts, _choice(rng, cfg["conversations"]["background_intents"]),
                               "background", rng.beta(1.5, 6))

        return {"customer_id": cid, "is_alive": alive, "dropout_ts": dropout_ts}

    # ------------------------------------------------------------------ run
    def expected_order_value(self, latent: pd.DataFrame) -> np.ndarray:
        """E[order value | latent], analytically: E[items] * sum_c affinity_c * E_c[price | tau]."""
        ocfg = self.cfg["orders"]["items_per_order"]
        mean_items = (ocfg["min"] + ocfg["max"]) / 2
        out = np.zeros(len(latent))
        for c in self.categories:
            _, prices = self._cat_products[c]
            e_price = np.array([(prices * self._product_weights(c, tau)).sum() for tau in latent["price_preference"]])
            out += latent[f"affinity_{c}"].to_numpy() * e_price
        return mean_items * out

    def run(self) -> dict[str, pd.DataFrame]:
        latent = self.sample_latent()
        status = pd.DataFrame([self.simulate_customer(r) for _, r in latent.iterrows()])
        latent = latent.merge(status, on="customer_id")
        latent["expected_annual_spend"] = latent["purchase_rate"] * 365 * self.expected_order_value(latent)

        events = pd.DataFrame(self.t.events).sort_values(["customer_id", "ts"], kind="stable")
        events.insert(0, "event_id", np.arange(1, len(events) + 1))
        for col in ("product_id", "order_id"):
            events[col] = events[col].astype("Int64")
        convs = pd.DataFrame(self.t.convs)
        convs["order_id"] = convs["order_id"].astype("Int64")
        orders = pd.DataFrame(self.t.orders)

        return {
            # observable
            "raw/customers": latent[["customer_id", "signup_date", "acquisition_channel"]],
            "raw/products": self.products,
            "raw/campaigns": self.campaigns,
            "raw/orders": orders,
            "raw/order_items": pd.DataFrame(self.t.items),
            "raw/events": events.reset_index(drop=True),
            "raw/conversations": convs,
            # hidden
            "ground_truth/customer_latent": latent,
            "ground_truth/conversation_labels": pd.DataFrame(self.t.conv_labels),
            "ground_truth/best_action": best_actions(latent, orders, self.snapshot, self.cfg["oracle"]),
        }


def generate(cfg: dict, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    tables = Simulator(cfg).run()
    out = Path(out_dir)
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return tables
