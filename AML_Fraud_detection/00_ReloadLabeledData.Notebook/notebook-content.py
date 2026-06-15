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

# CELL ********************

from pyspark.sql.functions import *
df_csv = (spark.read.option("header", True).option("inferSchema", False)
    .csv("Files/raw/transactions.csv"))
print("CSV rows:", df_csv.count())
print("Labels:", [r.pattern_label for r in df_csv.select("pattern_label").distinct().collect()])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_existing = spark.table("transactions")
print("Existing rows:", df_existing.count())
df_existing.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

common = sorted(set(df_csv.columns) & set(df_existing.columns))
print("Common columns:", common)
df_to_append = df_csv.select(common)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_to_append.write.format("delta").mode("append").saveAsTable("transactions")
print("Total rows now:", spark.table("transactions").count())
spark.table("transactions").groupBy("pattern_label").count().orderBy("count", ascending=False).show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for name in ["accounts","customers","watchlists","historical_cases","detection_rules"]:
    df = spark.read.option("header",True).option("inferSchema",False).csv(f"Files/raw/{name}.csv")
    df.write.format("delta").mode("overwrite").saveAsTable(name)
    print(f"  {name}: {df.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Layer 3 — Update detection_rules table:

from pyspark.sql import Row
new_rule = spark.createDataFrame([Row(
    rule_id="RULE-011", pattern_type="CIRCULAR_FLOW",
    detection_layer="KQL+ML", 
    condition_summary="Customer sends WIRE_OUT and receives WIRE_IN of 70-130% amount from same or linked counterparty within 30-365 days",
    default_severity="CRITICAL",
    rationale="Round-tripping / circular flow. FATF Red Flag for trade-based money laundering. Money leaves and returns to same entity through intermediaries."
)])
new_rule.write.format("delta").mode("append").saveAsTable("detection_rules")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
