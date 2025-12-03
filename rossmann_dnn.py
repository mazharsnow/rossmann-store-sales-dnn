import os
import calendar
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Reproducibility
# -----------------------------
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# RMSPE metric + numpy version
# -----------------------------
def rmspe_keras(y_true, y_pred):
    """
    Metric: y_true and y_pred are log1p(Sales).
    Convert back to Sales, then compute RMSPE.
    """
    y_true = tf.exp(y_true) - 1.0
    y_pred = tf.exp(y_pred) - 1.0

    epsilon = tf.constant(1e-6, dtype=tf.float32)
    y_true = tf.maximum(y_true, epsilon)

    pct_error = (y_pred - y_true) / y_true
    return tf.sqrt(tf.reduce_mean(tf.square(pct_error)))


def rmspe_numpy(y_true_log, y_pred_log):
    """
    Numpy RMSPE for printing after training.
    y_*_log are log1p(Sales).
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = y_true != 0
    return np.sqrt(np.mean(np.square((y_pred[mask] - y_true[mask]) / y_true[mask])))


# -----------------------------
# Feature engineering helpers
# -----------------------------
def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df


def add_competition_features(df):
    df["HasCompetition"] = (~df["CompetitionDistance"].isna()).astype(int)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(-1)

    def _months_since_competition_open(row):
        if pd.isna(row["CompetitionOpenSinceYear"]) or pd.isna(row["CompetitionOpenSinceMonth"]):
            return 0
        try:
            open_date = datetime(
                int(row["CompetitionOpenSinceYear"]),
                int(row["CompetitionOpenSinceMonth"]),
                1
            )
        except ValueError:
            return 0

        if pd.isna(row["Date"]):
            return 0

        months = (row["Date"].year - open_date.year) * 12 + (row["Date"].month - open_date.month)
        return max(0, months)

    df["CompetitionOpenMonths"] = df.apply(_months_since_competition_open, axis=1)
    return df


def add_promo2_features(df):
    def _weeks_since_promo2(row):
        if row["Promo2"] != 1:
            return 0
        if pd.isna(row["Promo2SinceYear"]) or pd.isna(row["Promo2SinceWeek"]):
            return 0
        try:
            year = int(row["Promo2SinceYear"])
            week = int(row["Promo2SinceWeek"])
            start = datetime.strptime(f"{year}-{week}-1", "%Y-%W-%w")
        except Exception:
            return 0

        if pd.isna(row["Date"]):
            return 0
        weeks = (row["Date"] - start).days // 7
        return max(0, weeks)

    df["Promo2OpenWeeks"] = df.apply(_weeks_since_promo2, axis=1)

    month_abbr = list(calendar.month_abbr)
    month_map = {i: month_abbr[i] for i in range(1, 13)}

    def _is_promo_month(row):
        if pd.isna(row["PromoInterval"]):
            return 0
        if "Month" not in row or pd.isna(row["Month"]):
            return 0
        month_str = month_map.get(int(row["Month"]), "")
        intervals = str(row["PromoInterval"]).split(",")
        return 1 if month_str in intervals else 0

    df["IsPromoMonth"] = df.apply(_is_promo_month, axis=1)
    return df


def add_lag_features(train_df):
    """
    Add per-store lag and rolling features based on Sales.
    Only applied to the training data (we don't know test Sales).
    """
    train_df = train_df.sort_values(["Store", "Date"])
    group = train_df.groupby("Store", group_keys=False)

    # Simple lags
    train_df["Sales_lag_1"] = group["Sales"].shift(1)
    train_df["Sales_lag_7"] = group["Sales"].shift(7)
    train_df["Sales_lag_14"] = group["Sales"].shift(14)

    # Rolling means (shifted to avoid leakage)
    train_df["Sales_rollmean_7"] = (
        group["Sales"].shift(1).rolling(window=7, min_periods=1).mean()
    )
    train_df["Sales_rollmean_30"] = (
        group["Sales"].shift(1).rolling(window=30, min_periods=1).mean()
    )

    return train_df


def add_store_aggregate_features(train_df, test_df):
    """
    Add store-level and store+day-of-week average sales features.
    Uses ONLY train sales, then merges into both train and test.
    """
    store_aggs = (
        train_df.groupby("Store")["Sales"]
        .agg(StoreAvgSales="mean", StoreMedianSales="median")
        .reset_index()
    )

    store_dow_aggs = (
        train_df.groupby(["Store", "DayOfWeek"])["Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Sales": "StoreDowAvgSales"})
    )

    train_df = train_df.merge(store_aggs, on="Store", how="left")
    test_df = test_df.merge(store_aggs, on="Store", how="left")

    train_df = train_df.merge(store_dow_aggs, on=["Store", "DayOfWeek"], how="left")
    test_df = test_df.merge(store_dow_aggs, on=["Store", "DayOfWeek"], how="left")

    return train_df, test_df


# -----------------------------
# Model builder
# -----------------------------
def build_dnn(input_dim):
    inputs = keras.Input(shape=(input_dim,), name="inputs")

    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",              # still MSE on log1p(Sales)
        metrics=[rmspe_keras],   # track RMSPE in original space
    )
    return model


# -----------------------------
# Main training + submission pipeline
# -----------------------------
def main():
    set_seeds(42)

    data_dir = "data"
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    store_path = os.path.join(data_dir, "store.csv")

    print("Loading data...")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)

    train["StateHoliday"] = train["StateHoliday"].astype(str)
    test["StateHoliday"] = test["StateHoliday"].astype(str)

    train = pd.merge(train, store, on="Store", how="left")
    test = pd.merge(test, store, on="Store", how="left")

    test["Open"] = test["Open"].fillna(1)

    print("Adding date features...")
    train = add_date_features(train)
    test = add_date_features(test)

    print("Adding competition features...")
    train = add_competition_features(train)
    test = add_competition_features(test)

    print("Adding Promo2 features...")
    train = add_promo2_features(train)
    test = add_promo2_features(test)

    # Add lag features BEFORE filtering Sales>0/Open
    print("Adding lag features...")
    train = add_lag_features(train)

    if "Customers" in train.columns:
        train = train.drop(columns=["Customers"])

    # Filter train: keep open days with positive sales (as before)
    train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()

    # Fill remaining NaNs in lag features with column medians
    lag_cols = [
        "Sales_lag_1",
        "Sales_lag_7",
        "Sales_lag_14",
        "Sales_rollmean_7",
        "Sales_rollmean_30",
    ]
    lag_medians = train[lag_cols].median()
    train[lag_cols] = train[lag_cols].fillna(lag_medians)

    print("Adding store aggregate features...")
    train, test = add_store_aggregate_features(train, test)

    # For test, we don't know true lags; approximate using StoreAvgSales
    for col in lag_cols:
        if col not in test.columns:
            test[col] = np.nan
        test[col] = test[col].fillna(test["StoreAvgSales"])

    # Define feature columns
    numeric_features = [
        "CompetitionDistance",
        "CompetitionOpenMonths",
        "Promo2OpenWeeks",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "StoreAvgSales",
        "StoreMedianSales",
        "StoreDowAvgSales",
        "Sales_lag_1",
        "Sales_lag_7",
        "Sales_lag_14",
        "Sales_rollmean_7",
        "Sales_rollmean_30",
    ]

    categorical_features = [
        "Store",
        "DayOfWeek",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "Promo",
        "Promo2",
        "Open",
        "HasCompetition",
        "IsPromoMonth",
    ]

    feature_cols = numeric_features + categorical_features

    split_date = pd.Timestamp(2015, 6, 15)
    train_df = train[train["Date"] < split_date].copy()
    val_df = train[train["Date"] >= split_date].copy()

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    y_train = np.log1p(train_df["Sales"].values)
    y_val = np.log1p(val_df["Sales"].values)

    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    numeric_transformer = StandardScaler()

    try:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Fitting preprocessor on training data...")
    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_proc = preprocessor.transform(X_val).astype(np.float32)

    input_dim = X_train_proc.shape[1]
    print(f"Input dimension after preprocessing: {input_dim}")

    model = build_dnn(input_dim)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_rmspe_keras",
            patience=7,
            mode="min",
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_rmspe_keras",
            factor=0.5,
            patience=3,
            mode="min",
            min_lr=1e-5,
        ),
    ]

    print("Training model...")
    history = model.fit(
        X_train_proc,
        y_train,
        validation_data=(X_val_proc, y_val),
        epochs=60,
        batch_size=2048,
        callbacks=callbacks,
        verbose=2,
    )

    val_pred_log = model.predict(X_val_proc).ravel()
    val_rmspe = rmspe_numpy(y_val, val_pred_log)
    print(f"Validation RMSPE: {val_rmspe:.5f}")

    print("Preparing test data for prediction...")
    test_ids = test["Id"].values

    for col in feature_cols:
        if col not in test.columns:
            print(f"Column {col} not in test; filling with 0.")
            test[col] = 0

    X_test = test[feature_cols]
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    print("Predicting on test set...")
    test_pred_log = model.predict(X_test_proc).ravel()
    test_sales = np.expm1(test_pred_log)

    closed_mask = test["Open"] == 0
    test_sales[closed_mask] = 0
    test_sales = np.clip(test_sales, 0, None)

    submission = pd.DataFrame({"Id": test_ids, "Sales": test_sales})
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved Kaggle submission file to: {submission_path}")


if __name__ == "__main__":
    main()
