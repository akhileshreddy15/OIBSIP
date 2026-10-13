from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "ecommerce_customers.csv"
OUT = BASE / "outputs"
FEATURES = ["DaysSinceLastPurchase", "PurchaseFrequency", "TotalSpend"]

def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("Duplicate customer IDs:", df.CustomerID.duplicated().sum())
    print("\nDescriptive statistics:\n", df[FEATURES].describe().round(2))
    X = StandardScaler().fit_transform(df[FEATURES])

    ks = list(range(2, 9)); inertia = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertia.append(model.inertia_)
    plt.figure(figsize=(8,5)); plt.plot(ks, inertia, marker="o")
    plt.title("Elbow Method for K-Means"); plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia"); plt.tight_layout(); plt.savefig(OUT/"elbow_method.png", dpi=160); plt.close()

    df["Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    profile = df.groupby("Cluster").agg(
        Customers=("CustomerID","count"),
        AvgDaysSinceLastPurchase=("DaysSinceLastPurchase","mean"),
        AvgPurchaseFrequency=("PurchaseFrequency","mean"),
        AvgOrderValue=("AverageOrderValue","mean"),
        AvgTotalSpend=("TotalSpend","mean"),
        TotalRevenue=("TotalSpend","sum")).round(2)
    profile.to_csv(OUT/"cluster_profile.csv")
    plt.figure(figsize=(9,6))
    sns.scatterplot(data=df,x="PurchaseFrequency",y="TotalSpend",hue="Cluster",palette="tab10",s=55)
    plt.title("Customer Clusters: Frequency vs Total Spend"); plt.tight_layout()
    plt.savefig(OUT/"customer_clusters.png",dpi=160); plt.close()
    plt.figure(figsize=(8,5)); df["Cluster"].value_counts().sort_index().plot(kind="bar")
    plt.title("Customers per Cluster"); plt.xlabel("Cluster"); plt.ylabel("Number of customers"); plt.tight_layout()
    plt.savefig(OUT/"customers_per_cluster.png",dpi=160); plt.close()

    actions=[]
    for cid,row in profile.iterrows():
        if row.AvgTotalSpend >= profile.AvgTotalSpend.quantile(.75):
            action="Reward with loyalty benefits and personalised premium offers."
        elif row.AvgDaysSinceLastPurchase >= profile.AvgDaysSinceLastPurchase.median():
            action="Try a re-engagement campaign with a relevant reminder or discount."
        elif row.AvgPurchaseFrequency <= profile.AvgPurchaseFrequency.median():
            action="Encourage repeat purchases with onboarding offers and recommendations."
        else:
            action="Promote bundles, cross-sells and loyalty points."
        actions.append(f"Cluster {cid}: {action}")
    summary="CUSTOMER SEGMENTATION ANALYSIS\nCustomers analysed: "+str(len(df))+"\nClusters: 4\n\nCLUSTER PROFILE\n"+profile.to_string()+"\n\nINITIAL RECOMMENDATIONS\n"+"\n".join(actions)+"\n\nDATA NOTE: Synthetic demo data for practice; not real customer data."
    (OUT/"analysis_summary.txt").write_text(summary,encoding="utf-8")
    print("\nCluster profile:\n",profile)
    print("\nOutputs saved to",OUT)
    print("Note: dataset is synthetic demo data.")
if __name__ == "__main__":
    main()
