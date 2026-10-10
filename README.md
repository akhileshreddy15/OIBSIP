# OASIS INFOBYTE SIP — Retail Sales Data EDA

**Track:** Data Analytics  
**Task:** Level 1, Task 1 — EDA on Retail Sales Data  
**Tools:** Python, pandas, Matplotlib, Seaborn, Jupyter Notebook

## Project overview
This beginner-friendly project explores retail transactions to identify sales trends, customer demographics, best-selling products, category revenue, numerical relationships, and one additional insight based on payment method.

> **Dataset transparency:** `data/retail_sales.csv` is a reproducible, synthetic demonstration dataset generated for this project. It is not real company data. The charts and business recommendations are examples; do not represent them as real business findings.

## Folder structure

```text
OIBSIP/
└── DataAnalytics-L1-EDARetailSales/
    ├── data/
    │   └── retail_sales.csv
    ├── outputs/
    │   ├── 01_monthly_sales_trend.png
    │   ├── 02_quarterly_sales_trend.png
    │   ├── 03_age_group_distribution.png
    │   ├── 04_gender_breakdown.png
    │   ├── 05_top_10_products.png
    │   ├── 06_category_revenue.png
    │   ├── 07_correlation_heatmap.png
    │   ├── 08_payment_method_average_revenue.png
    │   ├── analysis_summary.txt
    │   └── supporting CSV summaries
    ├── retail_sales_analysis.py
    ├── Retail_Sales_EDA.ipynb
    ├── requirements.txt
    └── README.md
```

## Setup and run

### Option A — Run the Python script

1. Install Python 3.10 or later.
2. Open a terminal/command prompt in this project folder.
3. (Recommended) Create a virtual environment:
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python retail_sales_analysis.py`
6. Open the `outputs` folder to view the charts, CSV summaries, and text report.

### Option B — Run the notebook

1. Install dependencies using `pip install -r requirements.txt`.
2. Start Jupyter: `jupyter notebook` (or open the `.ipynb` in VS Code).
3. Open `Retail_Sales_EDA.ipynb` and run all cells from top to bottom.

## Analysis checklist (matches the Oasis Infobyte task card)

- [x] Load dataset; inspect shape, data types, missing values, and duplicate rows.
- [x] Numerical descriptive statistics: mean, median, mode, standard deviation, and other summary statistics.
- [x] Monthly and quarterly sales trends using line charts.
- [x] Customer age-group distribution and gender breakdown.
- [x] Top 10 best-selling products and category revenue bar chart.
- [x] Correlation heatmap for numerical variables.
- [x] Additional visualisation: average transaction revenue by payment method.
- [x] Written observations and conclusion/recommendations.

## Outputs
The script creates eight PNG charts, CSV tables for summaries, and `analysis_summary.txt`. It also prints an overview and observations in the terminal.

## Key definitions
- **Revenue:** `Quantity × UnitPrice` for each transaction.
- **Monthly/quarterly sales:** sum of transaction revenue grouped by calendar month/quarter.
- **Best-selling products:** ranked by total units sold, not by revenue.
- **Correlation:** describes linear association; it does not prove causation. Revenue will naturally relate to quantity and unit price because of its formula.

## GitHub submission
The task-list PDF specifies one public repository named **`OIBSIP`**. Place this project folder at:

`OIBSIP/DataAnalytics-L1-EDARetailSales/`

Commit the source code, notebook, README, dataset, and relevant outputs. Follow your cohort's submission form instructions.

## Demo video and LinkedIn checklist
The PDF's general workflow asks for a screen-recorded walkthrough. Begin with a 2-second title card showing your full name, assigned track, and task title. Then show the notebook/script running and open the charts/report. Post the demo on LinkedIn, tag Oasis Infobyte, and include `#oasisinfobyte` plus relevant hashtags. The task list also asks interns to leave substantive comments on at least two other cohort demo videos.

Before submitting, replace the title-card placeholder with your own full name and verify the submission dates/form in your orientation email.
