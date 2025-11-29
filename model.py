
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


LOCAL_PATHS = {
    'historical': 'historical_data.csv',
    'fear_greed': 'fear_greed_index.csv'
}
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def try_read_csv(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {path} ({len(df)} rows, {len(df.columns)} cols)")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None

def to_datetime_safe(series, **kwargs):
    return pd.to_datetime(series, errors='coerce', **kwargs)

def find_first_contains(cols, tokens):
    for c in cols:
        low = c.lower()
        for t in tokens:
            if t in low:
                return c
    return None

def safe_get_column(df, col_names, default='unknown'):
    """Safely get a column from df trying multiple names, return as string series."""
    if isinstance(col_names, str):
        col_names = [col_names]
    for name in col_names:
        if name in df.columns:
            return df[name].astype(str)
    return default


def load_datasets():
    df_hist = try_read_csv(LOCAL_PATHS['historical'])
    df_fg = try_read_csv(LOCAL_PATHS['fear_greed'])
    if df_hist is None or df_fg is None:
        print("[FATAL] Both CSVs must be present. Edit LOCAL_PATHS if needed.")
        sys.exit(1)
    return df_hist, df_fg


def preprocess_and_merge(df_hist, df_fg):
    df_hist = df_hist.copy()
    df_fg = df_fg.copy()

    
    hist_cols = list(df_hist.columns)

    time_col = None
    for cand in ['time','timestamp','datetime','date','created_at']:
        if cand in [c.lower() for c in hist_cols]:
          
            for c in hist_cols:
                if c.lower() == cand:
                    time_col = c
                    break
            if time_col:
                break
    if time_col is None:
        time_col = find_first_contains(hist_cols, ['time','date','timestamp','datetime','created'])
    if time_col is None:
        print("[FATAL] No time/date column detected in historical CSV. Columns:", hist_cols)
        sys.exit(1)

   
    numeric_time = pd.to_numeric(df_hist[time_col], errors='coerce')
    if numeric_time.notna().sum() > len(df_hist) * 0.5: 
       
        med = numeric_time.median(skipna=True)
        if pd.notna(med):
            if med > 1e12:  
                df_hist['time'] = pd.to_datetime(numeric_time, unit='ms', errors='coerce')
                print(f"[PARSE] Parsed historical '{time_col}' as unix milliseconds.")
            elif med > 1e9:  # Likely seconds
                df_hist['time'] = pd.to_datetime(numeric_time, unit='s', errors='coerce')
                print(f"[PARSE] Parsed historical '{time_col}' as unix seconds.")
            else:  
                df_hist['time'] = to_datetime_safe(df_hist[time_col])
                print(f"[PARSE] Parsed historical '{time_col}' as datetime strings.")
        else:
            df_hist['time'] = to_datetime_safe(df_hist[time_col])
    else:
        
        df_hist['time'] = to_datetime_safe(df_hist[time_col])
        print(f"[PARSE] Parsed historical '{time_col}' as datetime strings.")
    
    before = len(df_hist)
    df_hist = df_hist[~df_hist['time'].isna()].copy()
    if len(df_hist) < before:
        print(f"[INFO] Dropped {before - len(df_hist)} rows with invalid timestamps in historical CSV.")

    
    df_hist['account'] = safe_get_column(df_hist, ['account', 'acct', 'user'], 'unknown_account')
    df_hist['symbol'] = safe_get_column(df_hist, ['symbol', 'pair'], 'unknown_symbol')
    
    # price candidates
    price_cand = None
    for c in df_hist.columns:
        if 'price' in c.lower() and 'execution' in c.lower():
            price_cand = c
            break
    if price_cand is None:
        price_cand = find_first_contains(df_hist.columns, ['price','exec_price','fill_price'])
    df_hist['execution_price'] = pd.to_numeric(df_hist.get(price_cand, np.nan), errors='coerce')
    
    size_cand = find_first_contains(df_hist.columns, ['size','qty','quantity','amount'])
    df_hist['size'] = pd.to_numeric(df_hist.get(size_cand, np.nan), errors='coerce')
    
    df_hist['side'] = safe_get_column(df_hist, ['side', 'direction'], 'unknown')
    
    pnl_cand = find_first_contains(df_hist.columns, ['closedpnl','closed_pnl','pnl','profit','pl','realized_pnl','realised_pnl'])
    df_hist['closedPnL'] = pd.to_numeric(df_hist.get(pnl_cand, np.nan), errors='coerce')
    
    lev_cand = find_first_contains(df_hist.columns, ['leverage','lev'])
    df_hist['leverage'] = pd.to_numeric(df_hist.get(lev_cand, np.nan), errors='coerce')


    df_hist['date'] = df_hist['time'].dt.date


    fg_cols = list(df_fg.columns)
  
    ts_col = None
    for c in fg_cols:
        if c.lower() == 'timestamp' or 'timestamp' in c.lower():
            ts_col = c
            break
    date_col = None
    for c in fg_cols:
        if c.lower() == 'date':
            date_col = c
            break

   
    df_fg['date_dt'] = pd.NaT
    if ts_col is not None:
      
        numeric_ts = pd.to_numeric(df_fg[ts_col], errors='coerce')
       
        med = numeric_ts.median(skipna=True)
        if pd.notna(med):
            if med > 1e12:
                unit = 'ms'
            else:
                unit = 's'
            try:
                df_fg['date_dt'] = pd.to_datetime(numeric_ts, unit=unit, errors='coerce')
                print(f"[PARSE] Parsed fear_greed '{ts_col}' as unix {unit} timestamps.")
            except Exception as e:
                print(f"[WARN] Failed to parse '{ts_col}' as unix-{unit}: {e}")

   
    if df_fg['date_dt'].notna().sum() < max(1, int(len(df_fg)*0.1)) and date_col is not None:
        try:
            df_fg['date_dt'] = to_datetime_safe(df_fg[date_col])
            print(f"[PARSE] Parsed fear_greed '{date_col}' as ISO dates.")
        except Exception:
            pass

   
    if df_fg['date_dt'].notna().sum() == 0:
        first_col = fg_cols[0] if len(fg_cols)>0 else None
        if first_col is not None:
            df_fg['date_dt'] = to_datetime_safe(df_fg[first_col].astype(str), dayfirst=False)
            print(f"[PARSE] Forced parse on first column '{first_col}', parsed {df_fg['date_dt'].notna().sum()} rows.")

   
    class_col = None
    for c in fg_cols:
        if 'class' in c.lower() or 'label' in c.lower() or 'sentiment' in c.lower():
            class_col = c
            break

    score_col = None
    for c in fg_cols:
        if 'value' in c.lower() or 'score' in c.lower() or 'index' in c.lower():
            score_col = c
            break

    if class_col is not None:
        df_fg['classification'] = df_fg[class_col].astype(str)
    elif score_col is not None:
        s = pd.to_numeric(df_fg[score_col], errors='coerce')
        bins = [-0.1,25,50,75,100.1]
        labels = ['Extreme Fear','Fear','Neutral','Greed']
        df_fg['classification'] = pd.cut(s.fillna(50), bins=bins, labels=labels)
    else:
       
        if 'classification' in df_fg.columns:
            df_fg['classification'] = df_fg['classification'].astype(str)
        else:
            df_fg['classification'] = 'Unknown'


    valid_before = len(df_fg)
    df_fg = df_fg.dropna(subset=['date_dt']).sort_values('date_dt').drop_duplicates(subset=['date_dt']).reset_index(drop=True)
    valid_after = len(df_fg)
    print(f"[INFO] fear_greed parsed dates: {valid_after} / {valid_before} kept (unique daily).")

    if valid_after == 0:
        print("[FATAL] Could not parse any dates in fear_greed CSV. Inspect the file format.")
        sys.exit(1)

    
    df_hist = df_hist.sort_values('time').reset_index(drop=True)
    df_fg = df_fg.sort_values('date_dt').reset_index(drop=True)
    
    
    print(f"[DIAG] Historical trades date range: {df_hist['time'].min()} to {df_hist['time'].max()}")
    print(f"[DIAG] Fear/Greed index date range: {df_fg['date_dt'].min()} to {df_fg['date_dt'].max()}")

    merged = pd.merge_asof(
        df_hist,
        df_fg[['date_dt','classification']].rename(columns={'date_dt':'fg_time'}),
        left_on='time',
        right_on='fg_time',
        direction='backward',
        tolerance=pd.Timedelta('7D') 
    )

    matched = merged['classification'].notna().sum()
    print(f"[MERGE] merge_asof matched {matched} / {len(merged)} trades (non-null classification).")

   
    if 'classification' not in merged.columns:
        merged['classification'] = np.nan
    merged['classification'] = merged['classification'].ffill().bfill().fillna('Unknown')

    
    unknown_frac = (merged['classification'] == 'Unknown').mean()
    print(f"[DIAG] Fraction Unknown after asof+fill: {unknown_frac:.3f}")
    if unknown_frac > 0.10:
        print("[FALLBACK] Applying weekly fallback mapping.")
        merged['week'] = merged['time'].dt.to_period('W').astype(str)
        df_fg['week'] = df_fg['date_dt'].dt.to_period('W').astype(str)
        weekly_map = df_fg.groupby('week')['classification'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').to_dict()
        merged['classification'] = merged.apply(
            lambda r: weekly_map.get(r['week'], r['classification']) if r['classification'] == 'Unknown' else r['classification'],
            axis=1
        )
        unknown_frac2 = (merged['classification']=='Unknown').mean()
        print(f"[FALLBACK] Fraction Unknown after weekly mapping: {unknown_frac2:.3f}")

    
    merged['is_profitable'] = (merged['closedPnL'] > 0).astype(int)
    merged['pnl_per_size'] = merged['closedPnL'] / merged['size'].replace({0: np.nan})

    
    print("[FINAL] Sentiment counts:", merged['classification'].value_counts().to_dict())
    print("[FINAL] Fraction Unknown:", float((merged['classification']=='Unknown').mean()))
    return merged


def exploratory_analysis(df):
    print("[INFO] Running exploratory analysis...")
    out = OUTPUT_DIR

    summary = df.groupby('classification').agg(
        trades=('account','count'),
        avg_pnl=('closedPnL','mean'),
        median_pnl=('closedPnL','median'),
        win_rate=('is_profitable','mean')
    ).sort_values('trades', ascending=False)
    summary.to_csv(os.path.join(out, 'summary_by_sentiment.csv'))
    print("[SAVED] summary_by_sentiment.csv")

    acc_summary = df.groupby('account').agg(
        total_trades=('account','count'),
        total_pnl=('closedPnL','sum'),
        avg_pnl=('closedPnL','mean'),
        win_rate=('is_profitable','mean')
    ).sort_values('total_pnl', ascending=False)
    acc_summary.to_csv(os.path.join(out, 'accounts_summary.csv'))
    print("[SAVED] accounts_summary.csv")

    # plots
    try:
        fig, ax = plt.subplots(figsize=(8,5))
        summary['win_rate'].plot(kind='bar', ax=ax)
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Market Sentiment')
        plt.tight_layout()
        fig.savefig(os.path.join(out, 'win_rate_by_sentiment.png'))
        plt.close(fig)
        print("[SAVED] win_rate_by_sentiment.png")
    except Exception as e:
        print("[WARN] Could not save win_rate plot:", e)

    try:
        fig, ax = plt.subplots(figsize=(8,5))
        summary['avg_pnl'].plot(kind='bar', ax=ax)
        ax.set_ylabel('Average PnL')
        ax.set_title('Average PnL by Market Sentiment')
        plt.tight_layout()
        fig.savefig(os.path.join(out, 'avg_pnl_by_sentiment.png'))
        plt.close(fig)
        print("[SAVED] avg_pnl_by_sentiment.png")
    except Exception as e:
        print("[WARN] Could not save avg_pnl plot:", e)

    # daily aggregation (use trade date)
    daily = df.groupby(df['time'].dt.date).agg(daily_pnl=('closedPnL','sum'), trades=('account','count')).reset_index().rename(columns={'time':'date'})
    daily.to_csv(os.path.join(out, 'daily_pnl.csv'), index=False)
    print("[SAVED] daily_pnl.csv")

    return summary, daily, acc_summary


def statistical_tests(df):
    print("[INFO] Running statistical tests...")
    out = OUTPUT_DIR
    counts = df['classification'].value_counts()
    if len(counts) < 2:
        msg = {"error": "Not enough sentiment groups to perform t-test", "present_groups": counts.to_dict()}
        pd.DataFrame([msg]).to_csv(os.path.join(out, 'stat_tests.csv'), index=False)
        print("[INFO] Statistical test skipped: fewer than 2 sentiment groups.")
        return msg

    top2 = counts.index[:2].tolist()
    a = df.loc[df['classification'] == top2[0], 'closedPnL'].dropna()
    b = df.loc[df['classification'] == top2[1], 'closedPnL'].dropna()
    if len(a) == 0 or len(b) == 0:
        msg = {"error": "Insufficient numeric PnL values in groups", "group1": top2[0], "n_a": len(a), "group2": top2[1], "n_b": len(b)}
        pd.DataFrame([msg]).to_csv(os.path.join(out, 'stat_tests.csv'), index=False)
        print("[INFO] Statistical test aborted: no numeric PnL in group(s).")
        return msg
    t_stat, p_val = ttest_ind(a, b, equal_var=False)
    res = {"group1": top2[0], "group2": top2[1], "t_stat": float(t_stat), "p_val": float(p_val), "n_a": int(len(a)), "n_b": int(len(b))}
    pd.DataFrame([res]).to_csv(os.path.join(out, 'stat_tests.csv'), index=False)
    print("[SAVED] stat_tests.csv")
    return res


def modeling(df):
    print("[INFO] Running simple classification model...")
    out = OUTPUT_DIR
    model_df = df.copy()
    model_df['size'] = pd.to_numeric(model_df.get('size', np.nan), errors='coerce')
    model_df['execution_price'] = pd.to_numeric(model_df.get('execution_price', np.nan), errors='coerce')
    model_df['leverage'] = pd.to_numeric(model_df.get('leverage', np.nan), errors='coerce')

    model_df['side'] = model_df.get('side','unknown').fillna('unknown').astype(str)
    model_df['classification'] = model_df.get('classification','Unknown').fillna('Unknown').astype(str)
    model_df['symbol'] = model_df.get('symbol','unknown').fillna('unknown').astype(str)

    y = model_df['is_profitable'].fillna(0).astype(int)
    X = model_df[['size','execution_price','leverage','side','classification','symbol']].copy()
    
    
    X['size'] = X['size'].fillna(X['size'].median())
    X['execution_price'] = X['execution_price'].fillna(X['execution_price'].median())
    X['leverage'] = X['leverage'].fillna(X['leverage'].median())
    
   
    X = X.dropna(how='all', subset=['size','execution_price','leverage'])
    y = y.loc[X.index]

    if len(X) < 50 or y.nunique() < 2:
        msg = f"Not enough data to train model: rows={len(X)}, unique_targets={y.nunique()}"
        with open(os.path.join(out, 'model_metrics.txt'), 'w') as f:
            f.write(msg + "\n")
        print("[INFO]", msg)
        return None, None, None

    X_enc = pd.get_dummies(X, columns=['side','classification','symbol'], drop_first=True)
    
    
    X_enc = X_enc.fillna(0)
    
    stratify = y if (y.nunique() > 1 and len(y) >= 50) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=stratify)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    with open(os.path.join(out, 'model_metrics.txt'), 'w') as f:
        f.write(f"Rows used for modeling: {len(X_enc)}\n")
        f.write(f"Train accuracy: {train_score}\n")
        f.write(f"Test accuracy: {test_score}\n")

    try:
        importances = pd.Series(clf.feature_importances_, index=X_enc.columns).sort_values(ascending=False).head(30)
        importances.to_csv(os.path.join(out, 'feature_importances.csv'))
        print("[SAVED] feature_importances.csv")
    except Exception as e:
        print("[WARN] Could not save feature importances:", e)

    print(f"[INFO] Model train acc: {train_score:.3f}, test acc: {test_score:.3f}")
    return clf, X_test, y_test


def cluster_accounts(df, n_clusters=5):
    print("[INFO] Clustering accounts...")
    out = OUTPUT_DIR
    agg = df.groupby('account').agg(total_trades=('account','count'), total_pnl=('closedPnL','sum'), win_rate=('is_profitable','mean')).replace([np.inf,-np.inf], np.nan).dropna()
    if len(agg) == 0:
        print("[WARN] No accounts to cluster.")
        agg.to_csv(os.path.join(out, 'accounts_clusters.csv'))
        return agg
    if len(agg) < n_clusters:
        n_clusters = max(1, len(agg)//2)
    scaler = StandardScaler()
    X = scaler.fit_transform(agg)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    agg['cluster'] = labels
    agg.to_csv(os.path.join(out, 'accounts_clusters.csv'))
    print("[SAVED] accounts_clusters.csv")
    return agg


def produce_report(summary, daily, acc_summary, stat_res):
    path = os.path.join(OUTPUT_DIR, 'report_summary.txt')
    with open(path, 'w') as f:
        f.write("Trader Behavior Insights - Report Summary\n")
        f.write("Generated: " + datetime.utcnow().isoformat() + "Z\n\n")
        f.write("1) Summary by Sentiment:\n")
        try:
            f.write(summary.to_string())
        except Exception:
            f.write("Could not stringify summary.\n")
        f.write("\n\n2) Statistical tests:\n")
        f.write(str(stat_res))
        f.write("\n\n3) Top accounts sample (top 10 by total_pnl):\n")
        try:
            f.write(acc_summary.head(10).to_string())
        except Exception:
            f.write("Could not stringify account summary.\n")
    print("[SAVED] report_summary.txt")


def main():
    try:
        df_hist, df_fg = load_datasets()
        merged = preprocess_and_merge(df_hist, df_fg)

        
        print("[DIAG] Sentiment counts after merge:", merged['classification'].value_counts().to_dict())
        print("[DIAG] Fraction Unknown:", float((merged['classification']=='Unknown').mean()))

        summary, daily, acc_summary = exploratory_analysis(merged)
        stat_res = statistical_tests(merged)
        clf, X_test, y_test = modeling(merged)
        clusters = cluster_accounts(merged, n_clusters=5)
        produce_report(summary, daily, acc_summary, stat_res)
        print("[DONE] All done. Check the 'outputs' directory for artifacts.")
    except Exception as e:
        print("[FATAL] Unexpected error:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()