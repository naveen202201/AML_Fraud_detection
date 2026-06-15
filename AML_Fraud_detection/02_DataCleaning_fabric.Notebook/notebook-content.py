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

# # 02_DataCleaning — Bronze → Silver
# Fixes vs old POC: correct `is_night` logic (old `between(22,6)` NEVER matched — hour 23 or 03 was
# not flagged), currency normalisation, dedupe across streaming appends, quarantine of bad rows.

# CELL ********************

from pyspark.sql.functions import *
from pyspark.sql.types import *

df_bronze = spark.table("transactions")

# Quarantine invalid rows instead of silently dropping (auditable in production)
bad = df_bronze.filter((col("amount") <= 0) | col("account_id").isNull() |
                       col("customer_id").isNull() | col("timestamp").isNull())
if bad.count() > 0:
    bad.withColumn("quarantined_at", current_timestamp()) \
       .write.format("delta").mode("append").saveAsTable("quarantine_transactions")
print("Quarantined rows:", bad.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# df_silver = (df_bronze
#     .dropDuplicates(["txn_id"])                       # Eventstream can deliver at-least-once
#     .filter((col("amount") > 0) & col("account_id").isNotNull()
#             & col("customer_id").isNotNull() & col("timestamp").isNotNull())
#     .withColumn("currency", coalesce(col("currency"), lit("INR")))
#     .withColumn("txn_timestamp", col("timestamp"))
#     .withColumn("txn_date", to_date("txn_timestamp"))
#     .withColumn("txn_hour", hour("txn_timestamp"))
#     .withColumn("txn_dayofweek", dayofweek("txn_timestamp"))
#     .withColumn("is_weekend", when(dayofweek("txn_timestamp").isin([1,7]), 1).otherwise(0))
#     # FIXED: night = 22:00-05:59. between(22,6) is an empty range and never fired in old POC.
#     .withColumn("is_night", when((col("txn_hour") >= 22) | (col("txn_hour") <= 5), 1).otherwise(0))
#     .withColumn("country_risk", when(col("country").isin(
#         ["NG","RU","AE","MX","PK","IR","KP","MM","KH","TR"]), 1).otherwise(0))   # FATF grey/black list
#     .withColumn("amount_bucket",
#         when(col("amount").between(8000, 9999), "NEAR_THRESHOLD")
#         .when(col("amount") > 100000, "VERY_LARGE")
#         .when(col("amount") > 10000, "LARGE")
#         .otherwise("NORMAL"))
#     .withColumn("is_cash", when(col("txn_type") == "CASH_DEP", 1).otherwise(0))
# )
# df_silver.write.format("delta").mode("overwrite").saveAsTable("silver_transactions")
# print("Silver:", df_silver.count(), "rows")
# df_silver.select("txn_id","amount","amount_bucket","is_night","country_risk").show(5)


df_bronze = spark.table("transactions")

# CAST timestamp string to proper timestamp type
df_bronze = df_bronze.withColumn("timestamp", to_timestamp(col("timestamp")))
# CAST amount string to double
df_bronze = df_bronze.withColumn("amount", col("amount").cast("double"))

df_silver = (df_bronze
    .dropDuplicates(["txn_id"])
    .filter((col("amount") > 0) & col("account_id").isNotNull()
            & col("customer_id").isNotNull() & col("timestamp").isNotNull())
    .withColumn("currency", coalesce(col("currency"), lit("INR")))
    .withColumn("txn_timestamp", col("timestamp"))
    .withColumn("txn_date", to_date("txn_timestamp"))
    .withColumn("txn_hour", hour("txn_timestamp"))
    .withColumn("txn_dayofweek", dayofweek("txn_timestamp"))
    .withColumn("is_weekend", when(dayofweek("txn_timestamp").isin([1,7]), 1).otherwise(0))
    .withColumn("is_night", when((col("txn_hour") >= 22) | (col("txn_hour") <= 5), 1).otherwise(0))
    .withColumn("country_risk", when(col("country").isin(
        ["NG","RU","AE","MX","PK","IR","KP","MM","KH","TR"]), 1).otherwise(0))
    .withColumn("amount_bucket",
        when(col("amount").between(8000, 9999), "NEAR_THRESHOLD")
        .when(col("amount") > 100000, "VERY_LARGE")
        .when(col("amount") > 10000, "LARGE")
        .otherwise("NORMAL"))
    .withColumn("is_cash", when(col("txn_type") == "CASH_DEP", 1).otherwise(0))
)
df_silver.write.format("delta").mode("overwrite").saveAsTable("silver_transactions")
print("Silver:", df_silver.count(), "rows")
df_silver.select("txn_id","amount","amount_bucket","is_night","country_risk").show(5)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# spark.table("transactions").groupBy("pattern_label").count().orderBy("count", ascending=False).show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# df_csv = (spark.read.option("header", True).option("inferSchema", False)
#     .csv("Files/raw/transactions.csv"))

# # Cast amount to double to match the Eventstream-created schema
# df_csv = df_csv.withColumn("amount", df_csv["amount"].cast("double"))

# # Match columns to the existing table
# existing_cols = spark.table("transactions").columns
# csv_cols = [c for c in existing_cols if c in df_csv.columns]
# df_csv.select(csv_cols).write.format("delta").mode("append").saveAsTable("transactions")

# # Verify
# spark.table("transactions").groupBy("pattern_label").count().orderBy("count", ascending=False).show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
