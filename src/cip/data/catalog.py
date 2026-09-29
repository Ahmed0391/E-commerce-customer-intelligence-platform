"""Product catalog and campaigns.

The catalog is deliberately simple: categories with distinct price levels and
templated names/descriptions. Descriptions are later embedded for RAG, so they
contain real, retrievable attributes (category, price band, use case), not filler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# category -> (median price EUR, product nouns, adjectives, use cases)
CATEGORIES: dict[str, tuple[float, list[str], list[str], list[str]]] = {
    "electronics": (120.0, ["Wireless Earbuds", "Smartwatch", "Bluetooth Speaker", "Power Bank", "USB-C Hub"],
                    ["Pro", "Lite", "Max", "Mini", "Plus"], ["commuting", "working from home", "travel"]),
    "home": (45.0, ["Ceramic Mug Set", "Linen Throw", "Desk Lamp", "Storage Basket", "Scented Candle"],
             ["Nordic", "Classic", "Soft", "Compact", "Rustic"], ["living room", "home office", "gifting"]),
    "fashion": (55.0, ["Denim Jacket", "Running Sneakers", "Wool Scarf", "Leather Belt", "Cotton Hoodie"],
                ["Slim", "Oversized", "Essential", "Premium", "Everyday"], ["casual wear", "winter", "weekend"]),
    "beauty": (25.0, ["Face Serum", "Hand Cream", "Lip Balm", "Shampoo Bar", "Sunscreen SPF50"],
               ["Hydrating", "Organic", "Sensitive", "Daily", "Repair"], ["daily routine", "dry skin", "summer"]),
    "sports": (60.0, ["Yoga Mat", "Resistance Bands", "Water Bottle", "Gym Bag", "Foam Roller"],
               ["Pro", "Travel", "Eco", "Grip", "Performance"], ["home workouts", "running", "recovery"]),
    "books": (18.0, ["Python Handbook", "Mystery Novel", "Cookbook", "Travel Guide", "Poetry Collection"],
              ["Illustrated", "Pocket", "Hardcover", "Annotated", "Bestselling"], ["learning", "leisure", "gifting"]),
    "toys": (30.0, ["Building Blocks", "Puzzle 1000pc", "Plush Bear", "Board Game", "RC Car"],
             ["Classic", "Deluxe", "Junior", "Family", "Mega"], ["family time", "birthdays", "education"]),
    "grocery": (12.0, ["Olive Oil", "Specialty Coffee", "Green Tea", "Dark Chocolate", "Honey Jar"],
                ["Organic", "Single-Origin", "Premium", "Fair-Trade", "Artisanal"], ["breakfast", "gifting", "cooking"]),
}

PRODUCTS_PER_CATEGORY = 25


def make_catalog(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    for cat, (median_price, nouns, adjs, uses) in CATEGORIES.items():
        for i in range(PRODUCTS_PER_CATEGORY):
            noun = nouns[i % len(nouns)]
            adj = adjs[(i // len(nouns)) % len(adjs)]
            price = float(np.round(median_price * rng.lognormal(0.0, 0.35), 2))
            use = uses[i % len(uses)]
            band = "budget" if price < 0.8 * median_price else "premium" if price > 1.25 * median_price else "mid-range"
            rows.append({
                "product_id": pid,
                "name": f"{adj} {noun}",
                "category": cat,
                "price": price,
                "price_band": band,
                "description": (f"{adj} {noun} in our {cat} range. A {band} option at EUR {price:.2f}, "
                                f"popular for {use}."),
            })
            pid += 1
    return pd.DataFrame(rows)


def make_campaigns(snapshot: pd.Timestamp) -> pd.DataFrame:
    """A handful of hand-specified campaigns active around the snapshot date.

    These become retrievable knowledge in milestone 6. They are fixed (not random)
    because they play the role of business decisions, not observations.
    """
    s = snapshot
    rows = [
        ("CMP-01", "Summer Electronics Week", "electronics", "all", 0.15, s - pd.Timedelta(days=10), s + pd.Timedelta(days=5)),
        ("CMP-02", "Welcome Back 20%", "all", "lapsed", 0.20, s - pd.Timedelta(days=60), s + pd.Timedelta(days=30)),
        ("CMP-03", "Loyalty Double Points", "all", "loyal", 0.0, s - pd.Timedelta(days=30), s + pd.Timedelta(days=30)),
        ("CMP-04", "Beauty Summer Bundle", "beauty", "all", 0.10, s - pd.Timedelta(days=20), s + pd.Timedelta(days=20)),
        ("CMP-05", "Free Express Shipping for New Customers", "all", "new", 0.0, s - pd.Timedelta(days=90), s + pd.Timedelta(days=60)),
        ("CMP-06", "Home Office Refresh", "home", "all", 0.12, s - pd.Timedelta(days=15), s + pd.Timedelta(days=15)),
    ]
    return pd.DataFrame(rows, columns=["campaign_id", "name", "category", "target_segment",
                                       "discount", "start_date", "end_date"])
