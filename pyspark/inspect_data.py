import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count

def main():
    spark = SparkSession.builder \
        .appName("Data Inspector") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("--- Loading /home/hduser/project/uk_road_accidents.csv ---")
    print("--- This will be SLOW as Spark is inferring the schema... ---")
    
    df = spark.read.csv(
       "/data/uk_accidents/uk_road_accidents.csv",
        header=True,
        inferSchema=True,
        escape="\""
    )
    
    # Drop unnamed first column (caused by leading comma)
    if "_c0" in df.columns:
        print("Dropping unnamed first column (_c0)...")
        df = df.drop("_c0")
    
    # Cache the data for faster operations
    df.cache()
    
    print("\n--- 1. INFERRED SCHEMA ---")
    df.printSchema()

    print("\n--- 2. FIRST 5 ROWS OF RAW DATA ---")
    df.show(5, truncate=False)
    
    total_count = df.count()
    print(f"\n--- 3. TOTAL ROW COUNT ---")
    print(f"Total rows in dataset: {total_count}")

    print("\n--- 4. NULL VALUE REPORT (FOR ALL COLUMNS) ---")
    # Check for null or NaN in every column
    null_checks = []
    for col_name, col_type in df.dtypes:
        if col_type in ('float', 'double'):
            null_checks.append(
                count(when(col(col_name).isNull() | isnan(col_name), col_name)).alias(col_name)
            )
        else:
            null_checks.append(
                count(when(col(col_name).isNull(), col_name)).alias(col_name)
            )
    
    df_null_report = df.select(null_checks)
    df_null_report.show(truncate=False, vertical=True)

    print("\n--- 5. UNIQUE VALUE COUNT FOR CATEGORICAL COLUMNS ---")

    # Based on your CSV header
    categorical_columns = [
        "Police_Force",
        "Accident_Severity",
        "Number_of_Vehicles",
        "Number_of_Casualties",
        "Day_of_Week",
        "Local_Authority_(District)",
        "Local_Authority_(Highway)",
        "1st_Road_Class",
        "Road_Type",
        "Speed_limit",
        "Junction_Control",
        "2nd_Road_Class",
        "Pedestrian_Crossing-Human_Control",
        "Pedestrian_Crossing-Physical_Facilities",
        "Light_Conditions",
        "Weather_Conditions",
        "Road_Surface_Conditions",
        "Special_Conditions_at_Site",
        "Carriageway_Hazards",
        "Urban_or_Rural_Area",
        "Did_Police_Officer_Attend_Scene_of_Accident"
    ]

    for col_name in categorical_columns:
        if col_name not in df.columns:
            print(f"\n⚠️ Skipping '{col_name}' — not found in dataset.")
            continue

        print(f"\n--- Analysis for: {col_name} ---")
        
        unique_count_df = df.groupBy(col_name).count()
        unique_count = unique_count_df.count()
        
        print(f"Total Unique Values: {unique_count}")
        print("Top 5 most common values:")
        unique_count_df.orderBy(col("count").desc()).show(5, truncate=False)

    print("\n--- ✅ Inspection Complete ---")
    spark.stop()

if __name__ == "__main__":
    main()
