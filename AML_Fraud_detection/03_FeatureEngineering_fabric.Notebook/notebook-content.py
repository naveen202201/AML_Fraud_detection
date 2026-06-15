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
# META     },
# META     "warehouse": {
# META       "known_warehouses": []
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 03_FeatureEngineering — Silver → Gold
# New vs old POC: **counterparty / flow features** (pass-through ratio, in-out time gap, distinct
# counterparties) — without these, mule and layering patterns are mathematically undetectable by the model.

# CELL ********************

from pyspark.sql.functions import *
from pyspark.sql.window import Window

df_silver    = spark.table("silver_transactions")
df_accounts  = spark.table("accounts")
df_customers = spark.table("customers")

w_acct_day  = Window.partitionBy("account_id","txn_date").orderBy("txn_timestamp") \
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
w_acct_week = Window.partitionBy("account_id").orderBy(col("txn_timestamp").cast("long")) \
                    .rangeBetween(-7*86400, 0)
w_acct_hour = Window.partitionBy("account_id").orderBy(col("txn_timestamp").cast("long")) \
                    .rangeBetween(-3600, 0)
w_acct_48h  = Window.partitionBy("account_id").orderBy(col("txn_timestamp").cast("long")) \
                    .rangeBetween(-48*3600, 0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cast numeric columns
df_silver = df_silver.withColumn("amount", col("amount").cast("double"))
df_accounts = (df_accounts
    .withColumn("days_inactive", col("days_inactive").cast("int"))
    .withColumn("txn_count_30d", col("txn_count_30d").cast("int"))
    .withColumn("avg_txn_amount", col("avg_txn_amount").cast("double"))
    .withColumn("countries_30d", col("countries_30d").cast("int"))
    .withColumn("cash_ratio", col("cash_ratio").cast("double"))
    .withColumn("risk_score", col("risk_score").cast("double"))
)
print("Casts done. Silver rows:", df_silver.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Flow features per account (mule / layering signals) ----
inflow  = (df_silver.filter(col("txn_type").isin("WIRE_IN"))
           .groupBy("account_id")
           .agg(sum("amount").alias("total_in"), max("txn_timestamp").alias("last_in"),
                countDistinct("counterparty_account").alias("distinct_senders")))
outflow = (df_silver.filter(col("txn_type").isin("WIRE_OUT","UPI_OUT"))
           .groupBy("account_id")
           .agg(sum("amount").alias("total_out"), min("txn_timestamp").alias("first_out_after"),
                countDistinct("counterparty_account").alias("distinct_receivers")))
flow = (inflow.join(outflow, "account_id", "outer").na.fill(0, ["total_in","total_out",
                                                                "distinct_senders","distinct_receivers"])
        .withColumn("pass_through_ratio",
                    when(col("total_in") > 0, col("total_out")/col("total_in")).otherwise(lit(0.0)))
        .withColumn("in_out_gap_min",
                    when(col("last_in").isNotNull() & col("first_out_after").isNotNull(),
                         (col("first_out_after").cast("long")-col("last_in").cast("long"))/60)
                    .otherwise(lit(None))))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_features = (df_silver
    .withColumn("txn_count_same_day", count("txn_id").over(w_acct_day))
    .withColumn("txn_count_7d",       count("txn_id").over(w_acct_week))
    .withColumn("txn_count_1h",       count("txn_id").over(w_acct_hour))
    .withColumn("amount_sum_1h",      sum("amount").over(w_acct_hour))
    .withColumn("cash_dep_count_48h", sum("is_cash").over(w_acct_48h))
    .withColumn("cash_sum_48h",       sum(when(col("is_cash")==1, col("amount")).otherwise(0)).over(w_acct_48h))
    .withColumn("near_threshold_flag", when(col("amount").between(8000, 9999), 1).otherwise(0))
    .join(df_accounts.select("account_id","avg_txn_amount","days_inactive","cash_ratio",
                             col("risk_score").alias("acct_risk_score")), "account_id", "left")
    .withColumn("amount_vs_avg", col("amount")/(coalesce(col("avg_txn_amount"), lit(0.0))+lit(0.01)))
    .withColumn("dormant_flag",  when(col("days_inactive") > 180, 1).otherwise(0))
    .join(df_customers.select("customer_id","pep_flag","watchlist_match",
                              col("risk_tier").alias("cust_risk_tier")), "customer_id", "left")
    .withColumn("pep_numeric",   when(col("pep_flag") == "Y", 1).otherwise(0))
    .withColumn("watchlist_hit", when(coalesce(col("watchlist_match"), lit("NONE")) != "NONE", 1).otherwise(0))
    .join(flow.select("account_id","pass_through_ratio","in_out_gap_min",
                      "distinct_senders","distinct_receivers"), "account_id", "left")
    .na.fill({"pass_through_ratio":0.0,"distinct_senders":0,"distinct_receivers":0})
    .withColumn("rapid_pass_through",
                when((col("pass_through_ratio") > 0.9) &
                     (coalesce(col("in_out_gap_min"), lit(99999)) <= 120), 1).otherwise(0))
    # label: only available on historical/training data; live records have pattern_label = PENDING
    .withColumn("is_fraud", when(col("pattern_label") == "CLEAN", 0)
                            .when(col("pattern_label") == "PENDING", lit(None).cast("int"))
                            .otherwise(1))
)
df_features.write.format("delta").mode("overwrite").saveAsTable("gold_features")
print("Gold:", df_features.count(), "rows,", len(df_features.columns), "columns")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
