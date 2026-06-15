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

# # 05_RealTimeScoring — score NEW streamed transactions every few minutes
# Loads the persisted model (no retraining), scores only transactions that arrived since the last
# run (`pattern_label = 'PENDING'` and not yet in `ml_risk_scores`), and raises flags.
# Schedule this notebook every 5 minutes via Data Factory pipeline — this is how ML scores are
# computed for "real-time" transactions in the Fabric-native architecture (micro-batch).
# KQL rules in the Eventhouse remain the true sub-5-second layer; this notebook adds the model score.

# CELL ********************

# %pip install scikit-learn joblib --quiet

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import joblib, uuid, pandas as pd
from pyspark.sql.functions import *

art = joblib.load("/lakehouse/default/Files/models/aml_model_v1.joblib")
scaler, iso, gbc, FEATURES = art["scaler"], art["iso"], art["gbc"], art["features"]

scored_ids = spark.table("ml_risk_scores").select("txn_id")
new_txn = (spark.table("gold_features")
           .join(scored_ids, "txn_id", "left_anti"))
n = new_txn.count()
print("Unscored transactions:", n)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if n > 0:
    pdf = new_txn.toPandas()
    X = pdf[FEATURES].fillna(0)
    raw = -iso.score_samples(scaler.transform(X))
    pdf["anomaly_score"] = ((raw - raw.min())/(raw.max()-raw.min()+1e-9)) if n > 1 else 0.5
    pdf["fraud_probability"] = gbc.predict_proba(X)[:,1]
    pdf["final_risk_score"] = (0.4*pdf["anomaly_score"] + 0.6*pdf["fraud_probability"]).clip(0,1)
    pdf["fraud_type"] = pdf["final_risk_score"].apply(lambda s: "UNKNOWN_SUSPICIOUS" if s > 0.75 else "CLEAN")
    pdf["is_fraud"] = None

    spark.createDataFrame(pdf[["txn_id","account_id","customer_id","amount","anomaly_score",
        "fraud_probability","final_risk_score","fraud_type","is_fraud"]]) \
        .withColumn("scored_at", current_timestamp()) \
        .write.format("delta").mode("append").saveAsTable("ml_risk_scores")

    hot = pdf[pdf["final_risk_score"] > 0.75].copy()
    if len(hot):
        hot["flag_id"] = [f"FLAG-RT-{uuid.uuid4().hex[:10]}" for _ in range(len(hot))]
        hot["rule_id"] = "RULE-010"; hot["detection_source"] = "ML_REALTIME"
        hot["severity"] = hot["final_risk_score"].apply(lambda s: "CRITICAL" if s>0.95 else "HIGH" if s>0.85 else "MEDIUM")
        hot["analyst_decision"] = "PENDING"
        hot["notes"] = "Real-time ML score " + hot["final_risk_score"].round(3).astype(str)
        hot = hot.rename(columns={"final_risk_score":"ml_score","fraud_type":"pattern_type"})
        spark.createDataFrame(hot[["flag_id","txn_id","account_id","customer_id","pattern_type",
                "rule_id","ml_score","detection_source","severity","analyst_decision","notes"]]) \
            .withColumn("created_at", current_timestamp()) \
            .select("flag_id","txn_id","account_id","customer_id","pattern_type","rule_id","ml_score",
                    "detection_source","severity","analyst_decision","created_at","notes") \
            .write.format("delta").mode("append").saveAsTable("suspicious_flags")
        print("Real-time flags raised:", len(hot))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
