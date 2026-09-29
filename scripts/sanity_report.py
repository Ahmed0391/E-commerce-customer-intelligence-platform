"""Check that the synthetic world has the structure we intend (and not structure we don't).

Each check is a question about a signal the pipeline will later rely on.
If a check fails, the data is wrong, and no downstream metric can be trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
rd = lambda n: pd.read_parquet(D / f"{n}.parquet")
cust, orders, events, convs = rd("raw/customers"), rd("raw/orders"), rd("raw/events"), rd("raw/conversations")
latent, labels, actions = rd("ground_truth/customer_latent"), rd("ground_truth/conversation_labels"), rd("ground_truth/best_action")
snapshot = pd.Timestamp("2026-06-30")

print("== Volumes ==")
n_orders = orders.groupby("customer_id").size().reindex(cust.customer_id, fill_value=0)
print(f"orders/customer: median {n_orders.median():.0f}, mean {n_orders.mean():.1f}, "
      f"zero-purchase {100*(n_orders==0).mean():.1f}%, max {n_orders.max()}")
print(f"true active at snapshot: {100*latent.is_alive.mean():.1f}%")
print(f"order value: median EUR {orders.order_value.median():.0f}, p90 EUR {orders.order_value.quantile(.9):.0f}")

print("\n== Q1: Does recency carry churn signal? (it should, imperfectly) ==")
last = orders.groupby("customer_id").order_ts.max()
buyers = latent.set_index("customer_id").loc[last.index]
recency = (snapshot - last).dt.days
auc = roc_auc_score(~buyers.is_alive, recency)
print(f"AUC(recency -> churned) among buyers = {auc:.3f}   (1.0 would mean trivially separable)")
# Relative recency: recency divided by the customer's own mean inter-purchase gap
gaps = orders.sort_values("order_ts").groupby("customer_id").order_ts.apply(lambda s: s.diff().dt.total_seconds().div(86400).mean()).clip(lower=1)
rel = (recency / gaps).dropna()
auc_rel = roc_auc_score(~buyers.loc[rel.index].is_alive, rel)
print(f"AUC(relative recency -> churned) (>=2 orders) = {auc_rel:.3f}  vs raw recency on same subset "
      f"{roc_auc_score(~buyers.loc[rel.index].is_alive, recency.loc[rel.index]):.3f}")

print("\n== Q2: Are complaints linked to churn? (they should be) ==")
issue_conv = set(convs.merge(labels, on="conversation_id").query("trigger=='delivery_issue'").customer_id)
buyers = buyers.assign(complained=buyers.index.isin(issue_conv))
print(buyers.groupby("complained").is_alive.agg(["mean", "size"]).rename(columns={"mean": "active_rate"}))

print("\n== Q3: Funnel (sessions reaching each stage) ==")
stage = events.groupby("session_id").event_type.agg(set)
for s in ["view", "add_to_cart", "checkout", "purchase"]:
    print(f"  {s:12s} {stage.apply(lambda x: s in x).mean()*100:5.1f}%")

print("\n== Q4: Intent and trigger distribution ==")
print(labels.groupby(["trigger", "intent"]).size().sort_values(ascending=False).head(12).to_string())

print("\n== Q5: Oracle action distribution ==")
print((actions.best_action.value_counts(normalize=True) * 100).round(1).to_string())

print("\n== Q6: Negative control: acquisition channel should NOT predict churn ==")
m = cust.merge(latent[["customer_id", "is_alive"]], on="customer_id")
print(m.groupby("acquisition_channel").is_alive.mean().round(3).to_string())
from scipy.stats import chi2_contingency
p = chi2_contingency(pd.crosstab(m.acquisition_channel, m.is_alive))[1]
print(f"chi-square independence test p = {p:.3f}  (channel is independent by construction; p < 0.05 would be a ~1-in-20 fluke)")

print("\n== Q7: Does purchased category reflect latent affinity? ==")
items = rd("raw/order_items").merge(rd("raw/products")[["product_id", "category"]], on="product_id") \
                             .merge(orders[["order_id", "customer_id"]], on="order_id")
obs_top = items.groupby("customer_id").category.agg(lambda s: s.value_counts().index[0])
agree = (obs_top == latent.set_index("customer_id").loc[obs_top.index, "top_category"]).mean()
print(f"most-bought category == latent top category: {100*agree:.1f}% of buyers")
