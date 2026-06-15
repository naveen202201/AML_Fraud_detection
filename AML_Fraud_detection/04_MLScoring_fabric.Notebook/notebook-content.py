# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "1fa23f09-4e4a-4a89-9395-085457bacf01",
# META       "default_lakehouse_name": "AMLLakehouse",
# META       "default_lakehouse_workspace_id": "7ffa9807-c3d9-4154-ba4e-853a2a6f4fcb",
# META       "known_lakehouses": [
# META         {
# META           "id": "1fa23f09-4e4a-4a89-9395-085457bacf01"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 04_MLScoring — Gold → ml_risk_scores + suspicious_flags
# Fixes vs old POC:
# 1. Model is **persisted** to `Files/models/` (old POC retrained from scratch every run and could not score real-time).
# 2. Trains ONLY on labelled rows, scores everything (old POC leaked: trained+scored on same 16 rows).
# 3. Flag append now matches `suspicious_flags` schema exactly (old append crashed: missing `notes`).
# 4. Unique flag IDs (UUID) — old `FLAG-ML-0,1,2...` collided on every rerun.

# CELL ********************

# %pip install scikit-learn joblib --quiet

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd, numpy as np, joblib, uuid, os
from pyspark.sql.functions import current_timestamp

df = spark.table("gold_features").toPandas()

FEATURES = ["amount","txn_count_same_day","txn_count_7d","txn_count_1h","amount_sum_1h",
            "cash_dep_count_48h","cash_sum_48h","near_threshold_flag","country_risk",
            "is_weekend","is_night","is_cash","dormant_flag","amount_vs_avg","pep_numeric",
            "watchlist_hit","pass_through_ratio","distinct_senders","distinct_receivers",
            "rapid_pass_through"]
X_all = df[FEATURES].fillna(0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- 1) Unsupervised anomaly model (catches UNKNOWN typologies) ----
scaler = StandardScaler().fit(X_all)
Xs = scaler.transform(X_all)
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42).fit(Xs)
raw = -iso.score_samples(Xs)
df["anomaly_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)   # normalise 0-1

# ---- 2) Supervised model on LABELLED rows only, proper holdout ----
lab = df[df["is_fraud"].notna()]
X_lab, y_lab = lab[FEATURES].fillna(0), lab["is_fraud"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X_lab, y_lab, test_size=0.25, stratify=y_lab, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
                                 random_state=42).fit(Xtr, ytr)
print("Holdout AUC:", round(roc_auc_score(yte, gbc.predict_proba(Xte)[:,1]), 4))
print(classification_report(yte, gbc.predict(Xte), digits=3))
df["fraud_probability"] = gbc.predict_proba(X_all)[:,1]

# ---- 3) Blended score ----
df["final_risk_score"] = (0.4*df["anomaly_score"] + 0.6*df["fraud_probability"]).clip(0, 1)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- 4) Persist model artefacts so the real-time scorer can reuse them ----
os.makedirs("/lakehouse/default/Files/models", exist_ok=True)
joblib.dump({"scaler": scaler, "iso": iso, "gbc": gbc, "features": FEATURES},
            "/lakehouse/default/Files/models/aml_model_v1.joblib")
print("Model saved to Files/models/aml_model_v1.joblib")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# # ---- 5) Rule-style typology labelling on scored output ----
# def fraud_type(r):
#     if r["near_threshold_flag"]==1 and r["txn_count_same_day"]>=3: return "STRUCTURING"
#     if r["rapid_pass_through"]==1 and r["distinct_senders"]>=2:    return "MULE_ACCOUNT"
#     if r["rapid_pass_through"]==1:                                 return "LAYERING"
#     if r["dormant_flag"]==1 and r["amount"]>50000:                 return "DORMANT_SPIKE"
#     if r["watchlist_hit"]==1 and r["amount"]>5000:                 return "WATCHLIST_HIT"
#     if r["pep_numeric"]==1 and r["amount"]>100000:                 return "PEP_LARGE_TXN"
#     if r["cash_dep_count_48h"]>=7 and r["is_cash"]==1:             return "SMURFING"
#     if r["is_cash"]==1 and r["is_night"]==1 and r["amount"]>25000: return "CASH_INTENSIVE"
#     if r["txn_count_1h"]>5:                                        return "HIGH_VELOCITY"
#     if r["final_risk_score"]>0.75:                                 return "UNKNOWN_SUSPICIOUS"
#     return "CLEAN"
# df["fraud_type"] = df.apply(fraud_type, axis=1)

# # ---- 6) Write ml_risk_scores ----
# out = spark.createDataFrame(df[["txn_id","account_id","customer_id","amount","anomaly_score",
#         "fraud_probability","final_risk_score","fraud_type","is_fraud"]]) \
#         .withColumn("scored_at", current_timestamp())
# out.write.format("delta").mode("overwrite").saveAsTable("ml_risk_scores")

from pyspark.sql.functions import col, current_timestamp
# ---- 5) Rule-style typology labelling on scored output ----
def fraud_type(r):
    if r["near_threshold_flag"]==1 and r["txn_count_same_day"]>=3: return "STRUCTURING"
    if r["rapid_pass_through"]==1 and r["distinct_senders"]>=2:    return "MULE_ACCOUNT"
    if r["rapid_pass_through"]==1:                                 return "LAYERING"
    if r["dormant_flag"]==1 and r["amount"]>50000:                 return "DORMANT_SPIKE"
    if r["watchlist_hit"]==1 and r["amount"]>5000:                 return "WATCHLIST_HIT"
    if r["pep_numeric"]==1 and r["amount"]>100000:                 return "PEP_LARGE_TXN"
    if r["cash_dep_count_48h"]>=7 and r["is_cash"]==1:             return "SMURFING"
    if r["is_cash"]==1 and r["is_night"]==1 and r["amount"]>25000: return "CASH_INTENSIVE"
    if r["txn_count_1h"]>5:                                        return "HIGH_VELOCITY"
    if r["final_risk_score"]>0.75:                                 return "UNKNOWN_SUSPICIOUS"
    return "CLEAN"
df["fraud_type"] = df.apply(fraud_type, axis=1)

# ---- 6) Write ml_risk_scores ----
out = spark.createDataFrame(df[["txn_id","account_id","customer_id","amount","anomaly_score",
        "fraud_probability","final_risk_score","fraud_type","is_fraud"]]) \
        .withColumn("is_fraud", col("is_fraud").cast("string")) \
        .withColumn("scored_at", current_timestamp())
out.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("ml_risk_scores")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# # ---- 7) Append NEW suspicious flags (full schema, no rerun duplicates) ----
# already = {r.txn_id for r in spark.table("suspicious_flags").select("txn_id").collect()}
# new = df[(df["final_risk_score"] > 0.75) & (~df["txn_id"].isin(already))].copy()
# new["flag_id"] = [f"FLAG-ML-{uuid.uuid4().hex[:10]}" for _ in range(len(new))]
# new["rule_id"] = "RULE-010"
# new["detection_source"] = "ML_NOTEBOOK"
# new["severity"] = new["final_risk_score"].apply(lambda s: "CRITICAL" if s > 0.95
#                                                 else "HIGH" if s > 0.85 else "MEDIUM")
# new["analyst_decision"] = "PENDING"
# new["notes"] = ("ML score " + new["final_risk_score"].round(3).astype(str)
#                 + " | typology " + new["fraud_type"])
# new = new.rename(columns={"final_risk_score":"ml_score","fraud_type":"pattern_type"})
# if len(new):
#     spark.createDataFrame(new[["flag_id","txn_id","account_id","customer_id","pattern_type",
#             "rule_id","ml_score","detection_source","severity","analyst_decision","notes"]]) \
#         .withColumn("created_at", current_timestamp()) \
#         .select("flag_id","txn_id","account_id","customer_id","pattern_type","rule_id","ml_score",
#                 "detection_source","severity","analyst_decision","created_at","notes") \
#         .write.format("delta").mode("append").saveAsTable("suspicious_flags")
# print(f"Scored {len(df)} txns | new flags appended: {len(new)}")


# ---- 7) Append NEW suspicious flags (full schema, no rerun duplicates) ----
from pyspark.sql.functions import col, current_timestamp, lit
import uuid

# Drop the old empty table with wrong schema
spark.sql("DROP TABLE IF EXISTS suspicious_flags")

already = set()  # table is gone, no existing flags
new = df[(df["final_risk_score"] > 0.75)].copy()
new["flag_id"] = [f"FLAG-ML-{uuid.uuid4().hex[:10]}" for _ in range(len(new))]
new["rule_id"] = "RULE-010"
new["detection_source"] = "ML_NOTEBOOK"
new["severity"] = new["final_risk_score"].apply(lambda s: "CRITICAL" if s > 0.95
                                                else "HIGH" if s > 0.85 else "MEDIUM")
new["analyst_decision"] = "PENDING"
new["notes"] = ("ML score " + new["final_risk_score"].round(3).astype(str)
                + " | typology " + new["fraud_type"])
new["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
new = new.rename(columns={"final_risk_score":"ml_score","fraud_type":"pattern_type"})
if len(new):
    spark.createDataFrame(new[["flag_id","txn_id","account_id","customer_id","pattern_type",
            "rule_id","ml_score","detection_source","severity","analyst_decision",
            "created_at","notes"]]) \
        .write.format("delta").mode("overwrite").saveAsTable("suspicious_flags")
print(f"Scored {len(df)} txns | new flags appended: {len(new)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.table("suspicious_flags").groupBy("pattern_type","severity").count().orderBy("severity","count", ascending=[True, False]).show(20)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# CIRCULAR FLOW DETECTION — matches KQL Rule 11 logic
from pyspark.sql.functions import *
import uuid, pandas as pd

txns = spark.table("gold_features").filter("txn_type IN ('WIRE_OUT','WIRE_IN')")

# Outbound: who sent money where
outbound = txns.filter("txn_type = 'WIRE_OUT'").filter("amount > 200000").select(
    col("account_id").alias("sender"),
    col("customer_id").alias("sender_cust"),
    col("counterparty_account").alias("receiver"),
    col("amount").alias("out_amount"),
    col("txn_timestamp").alias("out_time"),
    col("txn_id").alias("out_txn_id")
)

# Inbound: who received money from where
inbound = txns.filter("txn_type = 'WIRE_IN'").filter("amount > 200000").select(
    col("account_id").alias("recv_acct"),
    col("customer_id").alias("recv_cust"),
    col("counterparty_account").alias("from_acct"),
    col("amount").alias("in_amount"),
    col("txn_timestamp").alias("in_time"),
    col("txn_id").alias("in_txn_id")
)

# Find circles: A sends to B, then B sends back to A
circles = (outbound
    .join(inbound,
          (outbound.sender == inbound.from_acct) &
          (outbound.receiver == inbound.recv_acct), "inner")
    .filter("in_time > out_time")
    .filter("sender_cust != recv_cust")                           # Different customers only
    .withColumn("days_gap", datediff(col("in_time"), col("out_time")))
    .filter("days_gap BETWEEN 30 AND 365")                        # 30 days to 1 year
    .withColumn("return_ratio", round(col("in_amount") / col("out_amount"), 2))
    .filter("return_ratio BETWEEN 0.7 AND 1.3")                   # 70-130% return
)

circle_count = circles.count()
print(f"Circular flows found: {circle_count}")

if circle_count > 0:
    circles.select("sender", "sender_cust", "receiver", "recv_cust",
                   "out_amount", "in_amount", "return_ratio", "days_gap").show(20)

    # Write flags to suspicious_flags
    flags = circles.select(
        col("out_txn_id").alias("txn_id"),
        col("sender").alias("account_id"),
        col("sender_cust").alias("customer_id"),
        col("out_amount").alias("amount")
    ).toPandas()

    flags["flag_id"] = [f"FLAG-CIRC-{uuid.uuid4().hex[:10]}" for _ in range(len(flags))]
    flags["pattern_type"] = "CIRCULAR_FLOW"
    flags["rule_id"] = "RULE-011"
    flags["detection_source"] = "ML_NOTEBOOK"
    flags["severity"] = flags["amount"].apply(lambda a: "CRITICAL" if a > 1000000 else "HIGH")
    flags["ml_score"] = 0.95
    flags["analyst_decision"] = "PENDING"
    flags["notes"] = "Circular flow: money returned to sender within 1 year, different customer"
    flags["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    spark.createDataFrame(flags[["flag_id","txn_id","account_id","customer_id",
            "pattern_type","rule_id","ml_score","detection_source","severity",
            "analyst_decision","created_at","notes"]]) \
        .write.format("delta").mode("append") \
        .option("mergeSchema", "true").saveAsTable("suspicious_flags")
    print(f"CIRCULAR FLOW: {len(flags)} new flags written to suspicious_flags")
else:
    print("No circular flows detected — all clear")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
