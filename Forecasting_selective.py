# hgb_recent_month_forecast_recursive.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ====== EDIT THESE ======
BUYS_CSV  = Path("/audusd deals.csv")     # Settlement Date, Buy Amount
PRICE_CSV = Path("/Caustic_Soda_Weekly_Price.csv")
DAYFIRST = True
TRAIN_START = "2023-01-01"
VALID_WEEKS = 16
MAX_LAG_WEEKS = 8          # lags 1..MAX_LAG_WEEKS for both price & buys
FORECAST_WEEKS = 5         # ~one month ahead (5 Fridays)
LAG_DAYS = 6               # << NEW: shift buys BEHIND price by this many days in the history-only plot
# --- Tunables at the top of your file (near other EDIT THESE) ---
SMOOTH_WEEKS = 2
START_CUTOFF = "2023-01-01"

# ========================

# ---------- IO ----------
def read_buys(p: Path) -> pd.Series:
    to_float = lambda x: float(str(x).replace("$","").replace(",","").strip())
    df = pd.read_csv(p, parse_dates=["Settlement Date"], dayfirst=DAYFIRST)
    df.columns = [c.strip() for c in df.columns]
    df["Buy Amount"] = df["Buy Amount"].map(to_float)
    return df.groupby("Settlement Date")["Buy Amount"].sum().sort_index()

def read_price(p: Path) -> pd.Series:
    df = pd.read_csv(p, parse_dates=["Date"], dayfirst=DAYFIRST)
    df.columns = [c.strip() for c in df.columns]
    df["Last Price"] = pd.to_numeric(df["Last Price"], errors="coerce")
    return df.groupby("Date")["Last Price"].mean().sort_index()

def to_weekly(buys: pd.Series, price: pd.Series) -> pd.DataFrame:
    buys_w  = buys.resample("W-FRI").sum()
    price_w = price.resample("W-FRI").last().ffill()
    return pd.DataFrame({"buys": buys_w, "price": price_w}).dropna(subset=["price"])

def restrict_recent(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    return df.loc[pd.to_datetime(start_date):].copy()

# ---------- Features ----------
def add_lags_and_stats(df: pd.DataFrame, max_lag=MAX_LAG_WEEKS) -> pd.DataFrame:
    for L in range(1, max_lag + 1):
        df[f"price_lag_{L}"] = df["price"].shift(L)
        df[f"buys_lag_{L}"]  = df["buys"].shift(L)

    # Rolling stats (past-only)
    df["price_ma_4"]  = df["price"].shift(1).rolling(4, min_periods=2).mean()
    df["price_std_4"] = df["price"].shift(1).rolling(4, min_periods=2).std()
    df["buys_ma_4"]   = df["buys"].shift(1).rolling(4, min_periods=2).mean()
    df["buys_std_4"]  = df["buys"].shift(1).rolling(4, min_periods=2).std()

    # Changes
    df["price_wow"] = df["price"].pct_change()
    df["buys_wow"]  = df["buys"].pct_change().replace([np.inf, -np.inf], np.nan)

    # Calendar
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["month"]      = df.index.month.astype(int)
    df["quarter"]    = df.index.quarter.astype(int)
    df["year"]       = df.index.year.astype(int)
    return df

def build_training_frame(df_recent: pd.DataFrame, max_lag=MAX_LAG_WEEKS) -> pd.DataFrame:
    df = add_lags_and_stats(df_recent.copy(), max_lag=max_lag)
    df["y"] = np.log1p(df["buys"])
    return df.dropna()

def time_split(df_feat: pd.DataFrame, valid_weeks=VALID_WEEKS):
    if len(df_feat) <= valid_weeks + 10:
        valid_weeks = max(4, len(df_feat)//5)
    train = df_feat.iloc[:-valid_weeks].copy()
    valid = df_feat.iloc[-valid_weeks:].copy()
    return train, valid

# ---------- Model ----------
def train_hgb(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str]):
    X_tr, y_tr = train[feature_cols], train["y"]
    X_va, y_va = valid[feature_cols], valid["y"]
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=1500,
        max_depth=6,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=100,
        validation_fraction=min(0.25, len(X_va)/max(1, len(X_tr)+len(X_va))),
        random_state=42
    )
    model.fit(X_tr, y_tr)
    return model

def evaluate(model, df_feat: pd.DataFrame, feature_cols: list[str], label="Validation"):
    X = df_feat[feature_cols]
    y_true_log = df_feat["y"]
    y_pred_log = model.predict(X)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log).clip(0)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = (np.abs(y_pred - y_true) / np.maximum(1.0, y_true)).mean() * 100
    print(f"\n{label} metrics (orig scale): MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:,.2f}%")
    ax = pd.Series(y_true, index=df_feat.index).plot(figsize=(12,5), label=f"{label} actual", alpha=0.6)
    pd.Series(y_pred, index=df_feat.index).plot(ax=ax, label=f"{label} predicted", linewidth=2)
    ax.set_title(f"{label}: actual vs predicted weekly buys (recent regime)")
    ax.legend(); plt.tight_layout(); plt.show()

# ---------- Recursive forecast ----------
def make_feature_row(idx, price_combo, buys_combo, max_lag, template_cols):
    """Create one feature row for a single future index, using current combos."""
    row = {}
    row["price"] = price_combo.loc[idx]
    # lags
    for L in range(1, max_lag + 1):
        row[f"price_lag_{L}"] = price_combo.shift(L).reindex([idx]).iloc[0]
        row[f"buys_lag_{L}"]  = buys_combo.shift(L).reindex([idx]).iloc[0]
        # Note: reindex([idx]).iloc[0] is safe because we expand combos over future horizon

    # rolling stats (past-only, defined thanks to combo extension)
    row["price_ma_4"]  = price_combo.shift(1).rolling(4, min_periods=2).mean().reindex([idx]).iloc[0]
    row["price_std_4"] = price_combo.shift(1).rolling(4, min_periods=2).std().reindex([idx]).iloc[0]
    row["buys_ma_4"]   = buys_combo.shift(1).rolling(4, min_periods=2).mean().reindex([idx]).iloc[0]
    row["buys_std_4"]  = buys_combo.shift(1).rolling(4, min_periods=2).std().reindex([idx]).iloc[0]
    row["price_wow"]   = price_combo.pct_change().reindex([idx]).iloc[0]
    row["buys_wow"]    = buys_combo.pct_change().reindex([idx]).iloc[0]
    # calendar
    row_dt = pd.Timestamp(idx)
    row["weekofyear"] = row_dt.isocalendar().week
    row["month"]      = row_dt.month
    row["quarter"]    = row_dt.quarter
    row["year"]       = row_dt.year
    # assemble DataFrame with correct column order
    df_row = pd.DataFrame([row], index=[idx])
    # fill NA (first step may have NaNs)
    df_row = df_row.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    # ensure same columns/order as training
    df_row = df_row.reindex(columns=template_cols, fill_value=0.0)
    return df_row

def forecast_recursive(model, df_recent: pd.DataFrame, feature_cols: list[str],
                       max_lag=MAX_LAG_WEEKS, weeks=FORECAST_WEEKS):
    """
    Recursive multi-step forecast:
    - carry last observed weekly price forward (or replace with your own price forecast),
    - after predicting week t, append prediction to buys history
      so buys_lag_1 for t+1 is defined.
    """
    last_idx = df_recent.index[-1]
    future_idx = pd.date_range(last_idx + pd.offsets.Week(weekday=4), periods=weeks, freq="W-FRI")

    # Price: hold last observed price constant (replace with a forecast if you have one)
    price_forecast = pd.Series(df_recent["price"].iloc[-1], index=future_idx, name="price")
    price_combo = pd.concat([df_recent["price"], price_forecast])

    # Start buys_combo with history only; we’ll append predictions as we go.
    buys_combo = df_recent["buys"].copy()

    preds = []
    for t in future_idx:
        # Build one-row feature frame for this week using current combos
        X_row = make_feature_row(t, price_combo, buys_combo, max_lag, feature_cols)
        yhat_log = model.predict(X_row)[0]
        yhat = float(np.expm1(yhat_log))
        yhat = max(0.0, yhat)
        preds.append((t, yhat))
        # Append prediction to buys_combo so lags are available for next step
        buys_combo.loc[t] = yhat

    forecast = pd.Series(dict(preds), name="forecast_buys")
    return forecast

# ---------- Monthly aggregation ----------
def month_aggregate_from_weeks(weekly_series: pd.Series) -> pd.Series:
    df = weekly_series.to_frame("val")
    df["month"] = df.index.to_period("M").to_timestamp("M")
    return df.groupby("month")["val"].sum()

# ---------- NEW: Lagged history plot ----------
def plot_history_with_lag_offset(buys: pd.Series,
                                 price: pd.Series,
                                 lag_days: int,
                                 start_date: str = START_CUTOFF,
                                 smooth_weeks: int = SMOOTH_WEEKS,
                                 q_low: int = 10,
                                 q_high: int = 90):
    """
    Visual: Buys (shifted +lag_days) as bars; Price line shifted DOWN by a constant Δ.
    - We DO NOT scale price. We only:
        1) scale buys to roughly match price amplitude via quantile (q_low..q_high),
        2) compute Δ = mean(price - buys_scaled) and plot price - Δ.
    """

    if buys.empty or price.empty:
        return

    # 1) Trim to regime
    buys = buys.loc[pd.to_datetime(start_date):]
    price = price.loc[pd.to_datetime(start_date):]
    if buys.empty or price.empty:
        print(f"No data on/after {start_date}")
        return

    # 2) Shift buys BEHIND price by lag_days, align to weekly Fridays
    buys_d = buys.resample("D").sum()
    buys_d.index = buys_d.index + pd.Timedelta(days=lag_days)
    buys_w = buys_d.resample("W-FRI").sum()
    price_w = price.resample("W-FRI").last().ffill()

    # Optional light smoothing of buys
    if smooth_weeks and smooth_weeks > 0:
        buys_w = buys_w.rolling(smooth_weeks, min_periods=1).mean()

    aligned = pd.concat({"price": price_w, "buys": buys_w}, axis=1).dropna()
    if aligned.empty:
        print("No overlapping weekly data after shift.")
        return

    # 3) Robust amplitude match for buys using quantile range (IQR-like)
    p_span = np.percentile(aligned["price"], q_high) - np.percentile(aligned["price"], q_low)
    b_span = np.percentile(aligned["buys"], q_high) - np.percentile(aligned["buys"], q_low)
    scale = (p_span / b_span) if b_span > 0 else 1.0
    buys_scaled = aligned["buys"] * scale

    # 4) Constant vertical offset for the price line (move it DOWN)
    delta = (aligned["price"] - buys_scaled).mean()   # SSE-optimal constant shift
    price_shifted = aligned["price"] - delta

    # 5) Plot on the SAME axis so they truly hug
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(aligned.index, buys_scaled, width=5, alpha=0.35,
           label=f"Buys (shift +{lag_days}d, scaled)")
    ax.plot(aligned.index, price_shifted, linewidth=2,
            label=f"Price (shifted down {delta:.1f})")
    ax.set_title(f"Since {start_date}: Buys (shift +{lag_days}d) vs Price (offset only)")
    ax.set_ylabel("Shared visual scale")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

    print(f"Applied scale to buys: {scale:.4g}; shifted price down by Δ={delta:.2f}")


# ---------- Main ----------
def main():
    buys  = read_buys(BUYS_CSV)
    price = read_price(PRICE_CSV)

    # --- NEW: visualize history with lag that truly “hugs” the price ---
    # Visualize history with lag using constant offset (no price scaling)
    plot_history_with_lag_offset(buys, price, LAG_DAYS,
                                 start_date=START_CUTOFF,
                                 smooth_weeks=SMOOTH_WEEKS)

    df = to_weekly(buys, price)
    df_recent = restrict_recent(df, TRAIN_START)

    # Training frame (recent regime only)
    df_feat = build_training_frame(df_recent.copy(), max_lag=MAX_LAG_WEEKS)
    feature_cols = [c for c in df_feat.columns if c not in ["buys","y"]]

    train, valid = time_split(df_feat, VALID_WEEKS)
    print(f"Training rows: {len(train)} | Validation rows: {len(valid)}")
    print("Last history week:", df_recent.index[-1].date())
    model = train_hgb(train, valid, feature_cols)

    # Evaluate
    evaluate(model, train, feature_cols, "Training")
    evaluate(model, valid, feature_cols, "Validation")

    # Recursive weekly forecast
    weekly_fc = forecast_recursive(model, df_recent, feature_cols,
                                   max_lag=MAX_LAG_WEEKS, weeks=FORECAST_WEEKS)

    print("\n=== Weekly forecast (recursive, next ~month) ===")
    for dt, val in weekly_fc.items():
        print(f"{dt.date()}: {val:,.0f}")

    monthly = month_aggregate_from_weeks(weekly_fc)
    print("\n=== Aggregated forecast by calendar month ===")
    for m, val in monthly.items():
        print(f"{m.date()}: {val:,.0f}")

    # Plot forecast vs history (weekly)
    fig, ax = plt.subplots(figsize=(12,5))
    df_recent["buys"].plot(ax=ax, label="History (weekly buys)", alpha=0.6)
    weekly_fc.plot(ax=ax, label="Forecast (recursive)", marker="o", linewidth=2)
    cutoff = df_recent.index[-1] + pd.offsets.Week(weekday=4)
    ax.axvline(cutoff, linestyle="--", linewidth=1)
    ax.set_title("Weekly buys forecast — recent regime (recursive HGB)")
    ax.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
