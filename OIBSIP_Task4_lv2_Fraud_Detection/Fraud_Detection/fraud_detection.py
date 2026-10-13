from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
BASE=Path(__file__).resolve().parent
DATA=BASE/"data"/"transactions.csv"; OUT=BASE/"outputs"
def main():
    OUT.mkdir(exist_ok=True); df=pd.read_csv(DATA)
    print("Dataset shape:",df.shape,"\nMissing values:\n",df.isna().sum(),"\nClass counts:\n",df["Is_Fraud"].value_counts().sort_index())
    plt.figure(figsize=(6,4)); df["Is_Fraud"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0,1],["Legitimate (0)","Fraud (1)"],rotation=0); plt.ylabel("Number of transactions"); plt.title("Class Distribution"); plt.tight_layout(); plt.savefig(OUT/"class_distribution.png",dpi=160); plt.close()
    X=df.drop(columns=["Is_Fraud"]); y=df["Is_Fraud"]
    numeric=["Transaction_Amount_INR","Transaction_Hour","Days_Since_Last_Transaction","Account_Age_Months"]
    categorical=["Online_Transaction","International_Transaction","Merchant_Category"]
    prep=ColumnTransformer([("numeric",Pipeline([("imputer",SimpleImputer(strategy="median"))]),numeric),("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe=Pipeline([("preprocessor",prep),("model",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced",min_samples_leaf=2))])
    pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
    metrics={"Accuracy":accuracy_score(y_test,pred),"Precision_Fraud":precision_score(y_test,pred,zero_division=0),"Recall_Fraud":recall_score(y_test,pred,zero_division=0),"F1_Fraud":f1_score(y_test,pred,zero_division=0)}
    pd.DataFrame([metrics]).to_csv(OUT/"model_metrics.csv",index=False)
    report=classification_report(y_test,pred,zero_division=0); (OUT/"classification_report.txt").write_text(report,encoding="utf-8")
    print("\nMetrics:",metrics,"\nClassification report:\n",report)
    disp=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,pred,labels=[0,1]),display_labels=["Legitimate","Fraud"]); disp.plot(values_format="d"); plt.title("Confusion Matrix"); plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close()
    names=pipe.named_steps["preprocessor"].get_feature_names_out(); imp=pipe.named_steps["model"].feature_importances_
    fi=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=False); fi.to_csv(OUT/"feature_importance.csv",index=False)
    top=fi.head(12).sort_values("Importance"); plt.figure(figsize=(8,6)); plt.barh(top["Feature"],top["Importance"]); plt.xlabel("Random Forest feature importance"); plt.title("Top Feature Importances"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png",dpi=160); plt.close()
    summary=f"FRAUD DETECTION SUMMARY\nRows: {len(df)}\nFraud cases in dataset: {int(y.sum())}\nTrain rows: {len(X_train)}\nTest rows: {len(X_test)}\n\nMETRICS\n"+"\n".join(f"{k}: {v:.4f}" for k,v in metrics.items())+"\n\nDATA NOTE: Synthetic demo data only; not suitable for real fraud decisions.\n"
    (OUT/"analysis_summary.txt").write_text(summary,encoding="utf-8"); print("\nOutputs saved to:",OUT,"\nWarning: synthetic practice data only.")
if __name__=="__main__": main()
