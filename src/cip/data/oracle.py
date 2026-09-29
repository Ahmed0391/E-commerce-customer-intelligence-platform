"""Ground-truth 'best action' oracle.

The oracle sees the HIDDEN state (true activity status, true value, true purchase
rate). The recommendation system only sees observables. So the oracle defines what a
perfectly informed marketer would do, and the system is scored on how close it gets.
Some disagreement is irreducible: an observable-only system cannot know for certain
whether a quiet customer has really left (churned) or is just overdue (still active).

Rules, in priority order (a documented modelling choice, so version it):

  1. delivery problem in the last N days (any status)  -> service_recovery
  2. churned & high value                              -> winback_offer
  3. churned & not high value                          -> reactivation_email  (cheap)
  4. active & new                                      -> onboarding_nudge
  5. active & overdue & high value                     -> retention_incentive
  6. active & overdue & not high value                 -> reactivation_email
  7. active & high value                               -> loyalty_reward
  8. otherwise                                         -> cross_sell (true top category)

Why "overdue" and not "high dropout propensity"? Survivor bias: customers with high
dropout propensity and high purchase rate have mostly already churned by the snapshot,
so a propensity rule selects almost nobody. "Overdue" is defined from the Poisson model:
P(no purchase in r days | active) = exp(-lambda * r) < threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIONS = ["service_recovery", "winback_offer", "reactivation_email", "onboarding_nudge",
           "retention_incentive", "loyalty_reward", "cross_sell"]


def best_actions(latent: pd.DataFrame, orders: pd.DataFrame, snapshot: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    recent_cut = snapshot - pd.Timedelta(days=cfg["recent_issue_days"])
    recent_issue = set(orders.loc[(orders["delivery_status"] != "on_time")
                                  & (orders["delivery_date"] >= recent_cut)
                                  & (orders["delivery_date"] <= snapshot), "customer_id"])

    lat = latent.set_index("customer_id")
    last_purchase = orders.groupby("customer_id")["order_ts"].max().reindex(lat.index)
    last_activity = last_purchase.fillna(lat["signup_date"])          # no purchase yet -> since signup
    days_quiet = (snapshot - last_activity) / pd.Timedelta(days=1)
    p_quiet_if_active = np.exp(-lat["purchase_rate"] * days_quiet)
    overdue = p_quiet_if_active < cfg["overdue_survival_prob"]

    high_value = lat["expected_annual_spend"] >= lat["expected_annual_spend"].quantile(cfg["high_value_quantile"])
    is_new = lat["signup_date"] >= snapshot - pd.Timedelta(days=cfg["new_customer_days"])

    action = pd.Series("cross_sell", index=lat.index)
    # Assign from lowest to highest priority so higher-priority rules overwrite.
    action[high_value] = "loyalty_reward"
    action[overdue & ~high_value] = "reactivation_email"
    action[overdue & high_value] = "retention_incentive"
    action[is_new] = "onboarding_nudge"
    action[~lat["is_alive"] & ~high_value] = "reactivation_email"
    action[~lat["is_alive"] & high_value] = "winback_offer"
    action[lat.index.isin(recent_issue)] = "service_recovery"

    return pd.DataFrame({
        "customer_id": lat.index,
        "best_action": action.values,
        "target_category": lat["top_category"].values,
        "is_high_value": high_value.values,
        "is_overdue": (overdue & lat["is_alive"]).values,
    })
