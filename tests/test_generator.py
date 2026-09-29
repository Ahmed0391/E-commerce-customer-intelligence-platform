"""Invariants of the synthetic world. These protect the evaluation from silent data bugs."""
import copy

import pandas as pd
import pytest

from cip.data.conversations import fill_text
from cip.data.generator import Simulator, load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config("configs/generator.yaml")
    c["n_customers"] = 300
    return c


@pytest.fixture(scope="module")
def tables(cfg):
    return Simulator(copy.deepcopy(cfg)).run()


def test_reproducible(cfg, tables):
    again = Simulator(copy.deepcopy(cfg)).run()
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], again[name], obj=name)


def test_nothing_after_snapshot(cfg, tables):
    snap = pd.Timestamp(cfg["snapshot_date"])
    assert tables["raw/orders"]["order_ts"].max() <= snap
    assert tables["raw/events"]["ts"].max() <= snap
    assert tables["raw/conversations"]["ts"].max() <= snap


def test_no_purchases_after_dropout(tables):
    lat = tables["ground_truth/customer_latent"].set_index("customer_id")
    last = tables["raw/orders"].groupby("customer_id")["order_ts"].max()
    churned = lat.loc[~lat["is_alive"]]
    common = churned.index.intersection(last.index)
    # BG/NBD convention: dropout happens right after the last purchase
    assert (last.loc[common] == churned.loc[common, "dropout_ts"]).all()


def test_purchase_events_match_orders(tables):
    ev = tables["raw/events"]
    assert set(ev.loc[ev.event_type == "purchase", "order_id"]) == set(tables["raw/orders"]["order_id"])


def test_raw_tables_do_not_leak_labels(tables):
    leaky = {"intent", "trigger", "frustration", "is_alive", "dropout_ts", "purchase_rate",
             "dropout_prob", "best_action", "top_category", "expected_annual_spend"}
    for name, df in tables.items():
        if name.startswith("raw/"):
            assert not leaky & set(df.columns), f"{name} leaks {leaky & set(df.columns)}"


def test_recent_issue_gets_service_recovery(cfg, tables):
    snap = pd.Timestamp(cfg["snapshot_date"])
    o = tables["raw/orders"]
    recent = o[(o.delivery_status != "on_time")
               & (o.delivery_date >= snap - pd.Timedelta(days=cfg["oracle"]["recent_issue_days"]))
               & (o.delivery_date <= snap)].customer_id.unique()
    a = tables["ground_truth/best_action"].set_index("customer_id")
    assert (a.loc[recent, "best_action"] == "service_recovery").all()


def test_template_text_fill(tables):
    out = fill_text(tables["raw/conversations"], tables["ground_truth/conversation_labels"], source="templates")
    assert out["text"].notna().all()
    assert len(out) == len(tables["raw/conversations"])
