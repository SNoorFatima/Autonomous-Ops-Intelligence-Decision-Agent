import pandas as pd
import re

def normalize_col(c: str) -> str:
    c = c.strip()
    c = c.replace("\t", " ")          # your header has tabs
    c = re.sub(r"\s+", " ", c)
    c = c.replace(".", "_").replace("-", "_")
    c = c.replace("Sub_Category", "Sub_Category")
    return c

def load_and_clean(path_in: str, path_out: str):
    df = pd.read_csv(path_in, encoding="utf-8", engine="python")
    df.columns = [normalize_col(c) for c in df.columns]

    # Common renames across both datasets
    rename_map = {
        "Order_ID": "order_id",
        "Order ID": "order_id",
        "Order_Date": "order_date",
        "Order Date": "order_date",
        "Order_Date": "order_date",
        "Ship_Date": "ship_date",
        "Ship Date": "ship_date",
        "Ship_Mode": "ship_mode",
        "Ship Mode": "ship_mode",

        "Customer_ID Customer_Name": "customer_id",  # if it ever merges weirdly
        "Customer_ID": "customer_id",
        "Customer ID": "customer_id",
        "Customer_Name": "customer_name",
        "Customer Name": "customer_name",
        "Customer_Name": "customer_name",
        "Segment": "segment",

        "Country": "country",
        "City": "city",
        "State": "state",
        "Region": "region",
        "Market": "market",

        "Product_ID": "product_id",
        "Product ID": "product_id",
        "Product_Name": "product_name",
        "Product Name": "product_name",
        "Category": "category",
        "Sub_Category": "sub_category",
        "Sub-Category": "sub_category",

        "Sales": "sales",
        "Profit": "profit",
        "Quantity": "quantity",
        "Discount": "discount",
        "Shipping_Cost": "shipping_cost",
        "Shipping Cost": "shipping_cost",
    }

    # Apply renames only for columns that exist
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse dates if present
    for col in ["order_date", "ship_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Save cleaned
    df.to_csv(path_out, index=False)
    print("Saved:", path_out)
    print("Columns:", list(df.columns))

# Example usage:
# load_and_clean("data/global_superstore.csv", "data/global_superstore_clean.csv")
load_and_clean("data/train.csv", "data/superstore_us_clean.csv")