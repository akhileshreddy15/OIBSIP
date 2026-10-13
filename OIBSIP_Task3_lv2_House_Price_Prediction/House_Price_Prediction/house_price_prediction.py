from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "house_prices.csv"
OUT = BASE / "outputs"

def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDescriptive statistics:\n", df.describe(include="all").to_string())

    # Exploratory plots
    plt.figure(figsize=(8,5))
    plt.scatter(df["Area_sqft"], df["Price_INR"], alpha=.6)
    plt.xlabel("Area (sq ft)"); plt.ylabel("Price (INR)")
    plt.title("House Area vs Price"); plt.tight_layout()
    plt.savefig(OUT/"area_vs_price.png", dpi=160); plt.close()

    plt.figure(figsize=(8,6))
    numeric = df.select_dtypes(include="number")
    plt.imshow(numeric.corr(), aspect="auto")
    plt.xticks(range(len(numeric.columns)), numeric.columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric.columns)), numeric.columns)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap"); plt.tight_layout()
    plt.savefig(OUT/"correlation_heatmap.png", dpi=160); plt.close()

    target="Price_INR"
    features=["Area_sqft","Bedrooms","Age_years","Location","Quality_Rating"]
    X=df[features]; y=df[target]
    numeric_features=["Area_sqft","Bedrooms","Age_years","Quality_Rating"]
    categorical_features=["Location"]
    preprocessor=ColumnTransformer([
        ("numeric", Pipeline([("imputer",SimpleImputer(strategy="median"))]), numeric_features),
        ("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                  ("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
    ])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    models={
        "Linear Regression":LinearRegression(),
        "Ridge Regression":Ridge(alpha=1.0)
    }
    results=[]; fitted={}
    for name,model in models.items():
        pipe=Pipeline([("preprocessor",preprocessor),("model",model)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        rmse=mean_squared_error(y_test,pred)**0.5
        results.append({"Model":name,"MAE_INR":mean_absolute_error(y_test,pred),
                        "RMSE_INR":rmse,"R2":r2_score(y_test,pred)})
        fitted[name]=(pipe,pred)
    metrics=pd.DataFrame(results).round(3)
    metrics.to_csv(OUT/"model_metrics.csv",index=False)
    print("\nModel comparison:\n",metrics.to_string(index=False))

    best_name=metrics.sort_values("RMSE_INR").iloc[0]["Model"]
    best_pipe,best_pred=fitted[best_name]
    plt.figure(figsize=(7,6))
    plt.scatter(y_test,best_pred,alpha=.7)
    low=min(y_test.min(),best_pred.min()); high=max(y_test.max(),best_pred.max())
    plt.plot([low,high],[low,high],linestyle="--")
    plt.xlabel("Actual Price (INR)"); plt.ylabel("Predicted Price (INR)")
    plt.title(f"Actual vs Predicted — {best_name}"); plt.tight_layout()
    plt.savefig(OUT/"actual_vs_predicted.png",dpi=160); plt.close()

    residuals=y_test-best_pred
    plt.figure(figsize=(7,5))
    plt.scatter(best_pred,residuals,alpha=.7)
    plt.axhline(0,linestyle="--")
    plt.xlabel("Predicted Price (INR)"); plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot"); plt.tight_layout()
    plt.savefig(OUT/"residual_plot.png",dpi=160); plt.close()

    # Model coefficients after preprocessing
    feature_names=best_pipe.named_steps["preprocessor"].get_feature_names_out()
    coef=best_pipe.named_steps["model"].coef_
    coef_df=pd.DataFrame({"Feature":feature_names,"Coefficient_INR":coef})
    coef_df["AbsCoefficient"]=coef_df["Coefficient_INR"].abs()
    coef_df.sort_values("AbsCoefficient",ascending=False).to_csv(OUT/"feature_coefficients.csv",index=False)
    summary=(f"HOUSE PRICE PREDICTION SUMMARY\nRows analysed: {len(df)}\n"
             f"Training rows: {len(X_train)}\nTest rows: {len(X_test)}\nBest model by RMSE: {best_name}\n\n"
             f"MODEL METRICS\n{metrics.to_string(index=False)}\n\n"
             "DATA NOTE: Synthetic practice data; not actual market data.")
    (OUT/"analysis_summary.txt").write_text(summary,encoding="utf-8")
    print("\nBest model by RMSE:",best_name)
    print("Outputs saved to:",OUT)
    print("Dataset is synthetic demo data. Do not use predictions for real property decisions.")

if __name__=="__main__":
    main()
