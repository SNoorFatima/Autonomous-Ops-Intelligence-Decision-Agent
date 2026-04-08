import pandas as pd
import numpy as np

def analyze_supply_chain_kpis(csv_path: str) -> dict:
    """
    Reads your cleaned superstore CSV and returns a KPI summary your agent can use.
    Works with columns:
    order_date, ship_date, ship_mode, shipping_cost, region, category, sub_category,
    sales, profit, quantity, discount
    """
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist (safe checks)
    needed = ["order_date", "ship_date", "sales", "profit", "region", "category", "ship_mode"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}", "available_columns": list(df.columns)}

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # Shipping delay (days)
    df["ship_delay_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df = df.dropna(subset=["ship_delay_days", "sales", "profit", "region", "category", "ship_mode"])
    df = df[df["ship_delay_days"] >= 0]

    # Core KPIs
    total_sales = float(df["sales"].sum())
    total_profit = float(df["profit"].sum())
    profit_margin = float((total_profit / total_sales) * 100) if total_sales else 0.0

    avg_delay = float(df["ship_delay_days"].mean())
    p90_delay = float(np.percentile(df["ship_delay_days"], 90))

    # Top delay hotspots
    delay_by_region = (
        df.groupby("region")["ship_delay_days"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(2)
        .to_dict()
    )

    delay_by_shipmode = (
        df.groupby("ship_mode")["ship_delay_days"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    # Profit leakage (discount -> profit)
    # (Only if discount exists)
    discount_corr = None
    if "discount" in df.columns:
        try:
            discount_corr = float(df[["discount", "profit"]].corr().iloc[0, 1])
        except Exception:
            discount_corr = None

    # Shipping cost hotspots
    ship_cost = None
    if "shipping_cost" in df.columns:
        ship_cost = (
            df.groupby("ship_mode")["shipping_cost"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )

    return {
        "rows": int(len(df)),
        "total_sales": round(total_sales, 2),
        "total_profit": round(total_profit, 2),
        "profit_margin_%": round(profit_margin, 2),
        "avg_ship_delay_days": round(avg_delay, 2),
        "p90_ship_delay_days": round(p90_delay, 2),
        "top_delay_regions(avg_days)": delay_by_region,
        "delay_by_ship_mode(avg_days)": delay_by_shipmode,
        "avg_shipping_cost_by_ship_mode": ship_cost,
        "discount_profit_correlation": discount_corr
    }


if __name__ == "__main__":
    print(analyze_supply_chain_kpis("data/global_superstore_clean.csv"))