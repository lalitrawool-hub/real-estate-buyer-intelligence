# Real Estate Buyer Intelligence

A modular buyer segmentation and investment profiling starter dashboard for Parcl. The application uses the supplied `clients.csv` and `properties.csv` files and deliberately derives every feature from their observed columns. It does not assume buyer income, property valuations, coordinates, or any other unavailable field.

## Observed source fields

`clients.csv`: `client_id`, `client_type`, `first_name`, `last_name`, `date_of_birth`, `gender`, `country`, `region`, `acquisition_purpose`, `satisfaction_score`, `loan_applied`, `referral_channel`.

`properties.csv`: `listing_id`, `tower_number`, `transaction_date`, `unit_category`, `unit_number`, `floor_area_sqft`, `sale_price`, `listing_status`, `client_ref`.

Names and date of birth are not used. Buyer-level transaction count, mean recorded sale price, mean floor area, counts of observed categories/statuses, and days since last observed transaction are derived when their source columns exist. Sale price is parsed from currency-formatted text. Transactions are joined to clients using `client_ref` → `client_id`.

## Project structure

```text
app.py                 Streamlit dashboard and presentation layer
src/preprocessing.py   Loading, joining, feature aggregation, reusable encoding/scaling
src/clustering.py      K-Means and hierarchical experiments and evaluation metrics
src/profiling.py       Buyer segment summaries
clients.csv            Supplied client source data
properties.csv         Supplied property/transaction source data
```

## Install and run on Windows

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Keep both source CSV files beside `app.py`. The command above uses the detected per-user Python 3.12 installation. If that executable is missing, install Python 3.12 from [python.org](https://www.python.org/downloads/windows/), enable **Add python.exe to PATH** in the installer, reopen PowerShell, and replace the first two commands with `python --version` and `python -m venv .venv`. Python 3.12 is the recommended version for the available scientific Python wheels.

The sidebar filters country, region, acquisition purpose, and client type. Choose K-Means or hierarchical clustering and a segment count. Four sections show segmentation overview, investor behavior, geographic counts, and segment profiles. To start again later, open PowerShell in this folder and run `.venv\Scripts\python.exe -m streamlit run app.py`.

`.gitignore` excludes `.venv`, caches, secrets, and the supplied client and property CSVs because they may contain private information. Copy the two data files into the project folder locally before running the app.

## Modeling notes

The preprocessing transformer imputes numeric missing values with medians and categorical missing values with the most frequent observed category, then scales numeric data and one-hot encodes categories. Client IDs are excluded. K-Means has a fixed random seed for reproducibility. Internal metrics shown are silhouette (higher is better) and Davies–Bouldin (lower is better); they are unavailable when the data cannot support their calculation. Hierarchical clustering uses Ward linkage. Results are exploratory and will change with filter and model settings. No analytical result is stored or hard-coded.

Missing source files, empty files, missing join keys, unusable features, and insufficient filtered buyer counts produce clear messages rather than fabricated output.
