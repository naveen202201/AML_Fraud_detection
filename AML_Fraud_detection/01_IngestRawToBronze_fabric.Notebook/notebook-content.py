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

# # 01_IngestRawToBronze — Read CSVs from Lakehouse **Files/raw** and create Bronze Delta tables
# 
# **Production change vs old POC:** data is NO LONGER hardcoded in the notebook.
# Upload the 6 CSV files to `AMLLakehouse → Files → raw/` first:
# 
# ```
# Files/raw/transactions.csv      Files/raw/accounts.csv      Files/raw/customers.csv
# Files/raw/watchlists.csv        Files/raw/historical_cases.csv   Files/raw/detection_rules.csv
# ```
# 
# How to upload: open **AMLLakehouse → Files → ⋯ → New subfolder → `raw`** → open `raw` → **Upload → Upload files** → select all 6 CSVs.
# 
# This notebook validates each file, applies an explicit schema, and writes Bronze Delta tables.
# Attach **AMLLakehouse** to this notebook (left panel → Add lakehouse) before running.

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()
RAW = "Files/raw"          # relative path inside the attached Lakehouse
print("Spark ready:", spark.version)

# Fail fast if the raw folder is missing
files = [f.name for f in mssparkutils.fs.ls(RAW)]
required = ["transactions.csv","accounts.csv","customers.csv",
            "watchlists.csv","historical_cases.csv","detection_rules.csv"]
missing = [f for f in required if f not in files]
assert not missing, f"Missing files in Files/raw: {missing} — upload them first."
print("All raw files present:", files)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Explicit schemas — never infer schema in production (silent type drift breaks downstream jobs)

# CELL ********************

schema_txn = StructType([
    StructField("txn_id", StringType(), False),
    StructField("account_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("counterparty_account", StringType(), True),   # NEW: needed for mule/layering flow analysis
    StructField("amount", DoubleType(), False),
    StructField("currency", StringType(), True),               # NEW: multi-currency support
    StructField("txn_type", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("country", StringType(), True),
    StructField("merchant_cat", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("pattern_label", StringType(), True),          # ground-truth label (training only; absent in live feed)
])
schema_acc = StructType([
    StructField("account_id", StringType(), False), StructField("customer_id", StringType(), False),
    StructField("acct_type", StringType(), True),  StructField("open_date", DateType(), True),
    StructField("days_inactive", IntegerType(), True), StructField("txn_count_30d", IntegerType(), True),
    StructField("avg_txn_amount", DoubleType(), True), StructField("countries_30d", IntegerType(), True),
    StructField("cash_ratio", DoubleType(), True), StructField("risk_score", DoubleType(), True),
    StructField("risk_tier", StringType(), True),
])
schema_cust = StructType([
    StructField("customer_id", StringType(), False), StructField("name_hash", StringType(), True),
    StructField("country", StringType(), True), StructField("occupation", StringType(), True),
    StructField("pep_flag", StringType(), True), StructField("kyc_status", StringType(), True),
    StructField("risk_tier", StringType(), True), StructField("watchlist_match", StringType(), True),
    StructField("source_of_funds", StringType(), True),
])
schema_wl = StructType([
    StructField("entry_id", StringType(), False), StructField("entity_id", StringType(), True),
    StructField("entity_type", StringType(), True), StructField("list_name", StringType(), True),
    StructField("source", StringType(), True), StructField("risk_level", StringType(), True),
    StructField("added_date", DateType(), True), StructField("match_type", StringType(), True),
])
schema_case = StructType([
    StructField("case_id", StringType(), False), StructField("account_id", StringType(), True),
    StructField("customer_id", StringType(), True), StructField("pattern_type", StringType(), True),
    StructField("total_amount", DoubleType(), True), StructField("num_txns", IntegerType(), True),
    StructField("investigation_days", IntegerType(), True), StructField("outcome", StringType(), True),
    StructField("sar_filed", StringType(), True), StructField("close_date", DateType(), True),
])
schema_rules = StructType([
    StructField("rule_id", StringType(), False), StructField("pattern_type", StringType(), True),
    StructField("detection_layer", StringType(), True), StructField("condition_summary", StringType(), True),
    StructField("default_severity", StringType(), True), StructField("rationale", StringType(), True),
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def load(name, schema):
    df = (spark.read.option("header", True).option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
                .schema(schema).csv(f"{RAW}/{name}.csv"))
    return df

df_txn   = load("transactions", schema_txn)
df_acc   = load("accounts", schema_acc)
df_cust  = load("customers", schema_cust)
df_wl    = load("watchlists", schema_wl)
df_case  = load("historical_cases", schema_case)
df_rules = load("detection_rules", schema_rules)

# ---- Data-quality gates (fail fast, do not load bad data) ----
assert df_txn.count() >= 1000, "Expected >= 1000 transactions"
assert df_txn.filter(col("txn_id").isNull() | col("amount").isNull()).count() == 0, "Null keys/amounts found"
dupes = df_txn.groupBy("txn_id").count().filter("count > 1").count()
assert dupes == 0, f"{dupes} duplicate txn_ids in raw file"
orphans = df_txn.join(df_acc, "account_id", "left_anti").count()
print(f"Quality gates passed. Orphan account refs: {orphans} (counterparty/external accounts are expected)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Write Bronze Delta tables ----
df_txn.write.format("delta").mode("overwrite").saveAsTable("transactions")
df_acc.write.format("delta").mode("overwrite").saveAsTable("accounts")
df_cust.write.format("delta").mode("overwrite").saveAsTable("customers")
df_wl.write.format("delta").mode("overwrite").saveAsTable("watchlists")
df_case.write.format("delta").mode("overwrite").saveAsTable("historical_cases")
df_rules.write.format("delta").mode("overwrite").saveAsTable("detection_rules")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Create empty `suspicious_flags` and `ml_risk_scores` shells
# **Bug fixed from old POC:** the old seed schema for `suspicious_flags` had `pattern_id` + a `notes`
# column, but notebook 04 appended a dataframe WITHOUT `notes` and with fraud-type strings in
# `pattern_id` → Delta schema-mismatch append failure + inconsistent semantics.
# New schema uses `pattern_type` (human-readable) and 04 always writes every column.

# CELL ********************

schema_flag = StructType([
    StructField("flag_id", StringType(), False),
    StructField("txn_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("pattern_type", StringType(), True),      # STRUCTURING / LAYERING / ... (was pattern_id)
    StructField("rule_id", StringType(), True),           # NEW: traceability to detection_rules
    StructField("ml_score", DoubleType(), True),
    StructField("detection_source", StringType(), True),  # KQL_RULE / ML_NOTEBOOK / ONTOLOGY / RULE+ML
    StructField("severity", StringType(), True),
    StructField("analyst_decision", StringType(), True),  # PENDING / CONFIRMED / FALSE_POSITIVE
    StructField("created_at", TimestampType(), True),
    StructField("notes", StringType(), True),
])
spark.createDataFrame([], schema_flag).write.format("delta").mode("overwrite").saveAsTable("suspicious_flags")

schema_ml = StructType([
    StructField("txn_id", StringType(), False), StructField("account_id", StringType(), True),
    StructField("customer_id", StringType(), True), StructField("amount", DoubleType(), True),
    StructField("anomaly_score", DoubleType(), True), StructField("fraud_probability", DoubleType(), True),
    StructField("final_risk_score", DoubleType(), True), StructField("fraud_type", StringType(), True),
    StructField("is_fraud", IntegerType(), True), StructField("scored_at", TimestampType(), True),
])
spark.createDataFrame([], schema_ml).write.format("delta").mode("overwrite").saveAsTable("ml_risk_scores")

for t in ["transactions","accounts","customers","watchlists","historical_cases",
          "detection_rules","suspicious_flags","ml_risk_scores"]:
    print(f"  {t}: {spark.table(t).count()} rows")
print("ALL 8 BRONZE TABLES CREATED FROM Files/raw")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
