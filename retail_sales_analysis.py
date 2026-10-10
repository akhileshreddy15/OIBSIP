"""Oasis Infobyte SIP — Data Analytics L1: EDA on Retail Sales Data.
Run from this folder with: python retail_sales_analysis.py
Uses a clearly labelled synthetic dataset included in data/ so it works offline.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "retail_sales.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["AgeGroup"] = pd.cut(df["CustomerAge"], bins=[17, 24, 34, 44, 54, 64, 100],
                             labels=["18–24", "25–34", "35–44", "45–54", "55–64", "65+"])

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print("\nColumn data types:\n", df.dtypes.to_string())
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe().T
    desc["median"] = numeric.median()
    desc["mode"] = numeric.mode().iloc[0]
    desc.to_csv(OUT / "descriptive_statistics.csv")
    print("\n=== NUMERIC DESCRIPTIVE STATISTICS ===\n", desc.round(2).to_string())

    monthly = df.groupby("Month", as_index=False)["Revenue"].sum()
    quarterly = df.groupby("Quarter", as_index=False)["Revenue"].sum()
    monthly.to_csv(OUT / "monthly_sales.csv", index=False)
    quarterly.to_csv(OUT / "quarterly_sales.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="Month", y="Revenue", marker="o", ax=ax)
    ax.set(title="Monthly Sales Revenue Trend", xlabel="Month", ylabel="Revenue (₹)")
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    fig.savefig(OUT / "01_monthly_sales_trend.png", dpi=160); plt.close(fig)
    print("\nObservation — Monthly trend: highest month is", monthly.loc[monthly.Revenue.idxmax(), "Month"],
          "and lowest month is", monthly.loc[monthly.Revenue.idxmin(), "Month"],
          "based on the included sample data.")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=quarterly, x="Quarter", y="Revenue", marker="o", ax=ax)
    ax.set(title="Quarterly Sales Revenue Trend", xlabel="Quarter", ylabel="Revenue (₹)")
    ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    fig.savefig(OUT / "02_quarterly_sales_trend.png", dpi=160); plt.close(fig)
    print("Observation — Quarterly trend: compare the quarterly peaks and dips before planning inventory and promotions.")

    age_counts = df["AgeGroup"].value_counts().sort_index().rename_axis("AgeGroup").reset_index(name="Transactions")
    age_counts["AgeGroup"] = age_counts["AgeGroup"].astype(str)
    age_counts.to_csv(OUT / "age_group_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(age_counts["AgeGroup"], age_counts["Transactions"])
    ax.set(title="Customer Age-Group Distribution", xlabel="Age group", ylabel="Transactions")
    fig.tight_layout(); fig.savefig(OUT / "03_age_group_distribution.png", dpi=160); plt.close(fig)
    print("Observation — Age groups: the largest transaction count is in", age_counts.loc[age_counts.Transactions.idxmax(), "AgeGroup"], ".")

    gender = df["Gender"].value_counts().rename_axis("Gender").reset_index(name="Transactions")
    gender.to_csv(OUT / "gender_breakdown.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(gender["Gender"], gender["Transactions"])
    ax.set(title="Transactions by Gender", xlabel="Gender", ylabel="Transactions")
    fig.tight_layout(); fig.savefig(OUT / "04_gender_breakdown.png", dpi=160); plt.close(fig)
    print("Observation — Gender: use this breakdown to check whether campaign response differs by gender; do not assume causation.")

    top_products = df.groupby("Product", as_index=False).agg(UnitsSold=("Quantity", "sum"), Revenue=("Revenue", "sum"))
    top_products = top_products.sort_values("UnitsSold", ascending=False).head(10)
    top_products.to_csv(OUT / "top_10_products.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_products = top_products.sort_values("UnitsSold")
    ax.barh(plot_products["Product"], plot_products["UnitsSold"])
    ax.set(title="Top 10 Best-Selling Products by Units", xlabel="Units sold", ylabel="Product")
    fig.tight_layout(); fig.savefig(OUT / "05_top_10_products.png", dpi=160); plt.close(fig)
    print("Observation — Top product by units:", top_products.iloc[0]["Product"], "(", int(top_products.iloc[0]["UnitsSold"]), "units ).")

    category = df.groupby("Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    category.to_csv(OUT / "category_revenue.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(category["Category"], category["Revenue"])
    ax.set(title="Revenue by Product Category", xlabel="Category", ylabel="Revenue (₹)")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(OUT / "06_category_revenue.png", dpi=160); plt.close(fig)
    print("Observation — Revenue-leading category:", category.iloc[0]["Category"], ".")

    corr_cols = ["CustomerAge", "Quantity", "UnitPrice", "Revenue"]
    corr = df[corr_cols].corr(numeric_only=True)
    corr.to_csv(OUT / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation Heatmap of Numerical Variables")
    fig.tight_layout(); fig.savefig(OUT / "07_correlation_heatmap.png", dpi=160); plt.close(fig)
    print("Observation — Correlation: revenue is mechanically related to quantity and unit price because Revenue = Quantity × UnitPrice.")

    # Additional visualisation: average order revenue by payment method.
    payment = df.groupby("PaymentMethod", as_index=False).agg(Transactions=("TransactionID", "count"), AverageRevenue=("Revenue", "mean"))
    payment.to_csv(OUT / "payment_method_analysis.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(payment["PaymentMethod"], payment["AverageRevenue"])
    ax.set(title="Average Transaction Revenue by Payment Method", xlabel="Payment method", ylabel="Average revenue (₹)")
    fig.tight_layout(); fig.savefig(OUT / "08_payment_method_average_revenue.png", dpi=160); plt.close(fig)
    print("Observation — Payment methods: average transaction values differ by method in this sample; investigate convenience and customer mix before changing checkout options.")

    total_revenue = df["Revenue"].sum()
    total_units = df["Quantity"].sum()
    avg_transaction = df["Revenue"].mean()
    recommendations = [
        f"Prioritise stock availability and product-page visibility for {top_products.iloc[0]['Product']}, the leading product by units in this sample.",
        f"Review category-specific promotions for {category.iloc[0]['Category']}, which contributes the most sample revenue; test campaigns and compare results before scaling.",
        f"Plan monthly inventory and staffing around observed monthly peaks ({monthly.loc[monthly.Revenue.idxmax(), 'Month']}) and lows ({monthly.loc[monthly.Revenue.idxmin(), 'Month']}).",
        "Run a small, measurable campaign for underrepresented age groups and compare conversion and margin, rather than targeting on transaction counts alone."
    ]
    report = ["OASIS INFOBYTE SIP — RETAIL SALES EDA SUMMARY", "Dataset note: included data is synthetic demonstration data, not real company sales.",
              f"Transactions: {len(df):,}", f"Total units sold: {total_units:,}", f"Total revenue: ₹{total_revenue:,.2f}",
              f"Average transaction revenue: ₹{avg_transaction:,.2f}", f"Rows with missing values: {int(df.isna().any(axis=1).sum())}",
              f"Duplicate rows: {int(df.duplicated().sum())}", "", "ACTIONABLE RECOMMENDATIONS:"]
    report += [f"{i+1}. {rec}" for i, rec in enumerate(recommendations)]
    report += ["", "Limitations: results are illustrative because the dataset is synthetic. Validate insights with genuine, sufficiently complete sales data before making business decisions."]
    (OUT / "analysis_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nDone. Charts and CSV outputs saved to: {OUT}")

if __name__ == "__main__":
    main()
