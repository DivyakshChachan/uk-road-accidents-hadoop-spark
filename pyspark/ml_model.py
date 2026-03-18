from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoderEstimator, VectorAssembler, Imputer, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.types import DoubleType

def run_accident_model():
    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("AccidentSeverityGridSearch") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # 2. Load Data (From HDFS)
    print("--- Loading Data from HDFS ---")
    file_path = "/data/uk_accidents/uk_road_accidents.csv"
    
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True, escape="\"")
    except Exception as e:
        print(f"Error loading data: {e}")
        spark.stop()
        return

    # 3. Feature Engineering & Cleaning
    print("--- Cleaning & Feature Engineering ---")
    
    # Handle Missing Values (-1 codes)
    columns_to_clean = [
        "Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude",
        "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", 
        "Urban_or_Rural_Area", "Junction_Control", "Pedestrian_Crossing-Physical_Facilities", 
        "Special_Conditions_at_Site", "Carriageway_Hazards"
    ]
    
    for col_name in columns_to_clean:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.when(F.col(col_name) == -1, None).otherwise(F.col(col_name)))

    # Create Time Features
    df = df.withColumn("Accident_Date", F.to_date(F.trim(F.col("Date")), "dd/MM/yyyy"))
    df = df.withColumn("Month", F.month(F.col("Accident_Date")))
    
    # Extract Hour and cast strictly to Double
    df = df.withColumn("Time", F.when(F.col("Time").isNull(), "00:00").otherwise(F.col("Time"))) \
           .withColumn("Hour", F.split(F.col("Time"), ":")[0].cast("double"))

    # --- DEFINE SELECTED FEATURES ---
    label_col = "Accident_Severity"

    categorical_cols = [
        "Day_of_Week", "Road_Type", "Light_Conditions", 
        "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area",
        "1st_Road_Class", "Junction_Control", "Pedestrian_Crossing-Physical_Facilities",
        "Special_Conditions_at_Site", "Carriageway_Hazards", "Police_Force"
    ]
    
    numeric_cols = [
        "Longitude", "Latitude", "Number_of_Vehicles", "Speed_limit", 
        "Hour", "Month", "Year"
    ]

    # --- FIX: Cast all numeric columns to Double for Imputer ---
    # This solves the "must be double/float but was int" error
    print("--- Casting Numeric Columns to Double ---")
    for col_name in numeric_cols:
        # We use F.col(col_name).cast("double") to ensure Imputer accepts it
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast("double"))

    # Drop rows with nulls in these specific columns
    critical_cols = [label_col] + categorical_cols + numeric_cols
    # Ensure we only drop based on columns that actually exist
    existing_critical_cols = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=existing_critical_cols)

    # 4. Handle Class Imbalance (Smart Balancing)
    print("--- Balancing Dataset (Targeting 'Serious' Count) ---")
    df.cache()
    
    class_counts = df.groupBy("Accident_Severity").count().collect()
    counts_dict = {row['Accident_Severity']: row['count'] for row in class_counts}
    
    if not counts_dict:
        print("Error: Dataset appears empty.")
        spark.stop()
        return

    target_count = counts_dict.get(2, 10000) 
    
    count_fatal = counts_dict.get(1, 1)
    count_slight = counts_dict.get(3, 1)
    
    ratio_fatal = float(target_count) / count_fatal
    ratio_slight = float(target_count) / count_slight
    
    print(f"Class Counts: {counts_dict}")
    print(f"Oversample Fatal (1) by: {ratio_fatal:.2f}x")
    print(f"Undersample Slight (3) by: {ratio_slight:.2f}x")

    df_fatal = df.filter(F.col("Accident_Severity") == 1)
    df_serious = df.filter(F.col("Accident_Severity") == 2)
    df_slight = df.filter(F.col("Accident_Severity") == 3)

    df_fatal_bal = df_fatal.sample(withReplacement=True, fraction=ratio_fatal, seed=42)
    df_slight_bal = df_slight.sample(withReplacement=False, fraction=ratio_slight, seed=42)
    df_serious_bal = df_serious

    df_balanced = df_fatal_bal.unionAll(df_serious_bal).unionAll(df_slight_bal)
    
    # 5. Define ML Pipeline
    print("--- Building Pipeline ---")

    label_indexer = StringIndexer(inputCol="Accident_Severity", outputCol="label").setHandleInvalid("skip")
    
    # Numeric Handling
    num_imputer = Imputer(inputCols=numeric_cols, outputCols=[f"{c}_imputed" for c in numeric_cols]).setStrategy("mean")
    
    num_assembler = VectorAssembler(inputCols=num_imputer.getOutputCols(), outputCol="imputed_num_vector").setHandleInvalid("skip")
    
    num_scaler = StandardScaler(inputCol="imputed_num_vector", outputCol="scaled_num_features", withStd=True, withMean=True)

    # Categorical Handling
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx").setHandleInvalid("keep") for c in categorical_cols]
    
    ohe_input = [f"{c}_idx" for c in categorical_cols]
    ohe_output = [f"{c}_ohe" for c in categorical_cols]
    
    encoder = OneHotEncoderEstimator(inputCols=ohe_input, outputCols=ohe_output).setHandleInvalid("keep")

    # Final Assembly
    final_assembler = VectorAssembler(inputCols=["scaled_num_features"] + ohe_output, outputCol="features")

    # Base Random Forest
    # MODIFIED: Using fixed parameters (50 trees, depth 10) instead of GridSearch to save memory
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42, numTrees=50, maxDepth=10)

    pipeline = Pipeline(stages=[label_indexer, num_imputer, num_assembler, num_scaler] + indexers + [encoder, final_assembler, rf])

    # 6. Train (Single Model)
    print("--- Starting Single Model Training (No Grid Search) ---")

    # Evaluator (F1 Score)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    (trainingData, testData) = df_balanced.randomSplit([0.7, 0.3], seed=42)
    
    # Fit pipeline directly
    model = pipeline.fit(trainingData)
    
    print("--- Training Complete ---")

    # 7. Results
    print("--- Evaluating Model ---")
    
    predictions = model.transform(testData)
    
    f1 = evaluator.evaluate(predictions)
    print(f"Model Weighted F1 Score: {f1}")
    
    # Get the RF model from the trained pipeline
    trained_rf = model.stages[-1]
    
    print("\n" + "="*30)
    print(" MODEL PARAMETERS ")
    print("="*30)
    print(f"Num Trees: {trained_rf.getNumTrees}")
    print(f"Max Depth: {trained_rf.getOrDefault('maxDepth')}")
    print("="*30 + "\n")

    print("Confusion Matrix (Label vs. Prediction):")
    predictions.groupBy("label", "prediction").count().show()

    # --- Display Label Mapping ---
    print("\n" + "="*30)
    print(" LABEL MAPPING (0, 1, 2) ")
    print("="*30)
    
    # Get the fitted StringIndexer from the pipeline (Stage 0)
    label_indexer_model = model.stages[0]
    
    # Map original dataset values (1, 2, 3) to names
    # Note: StringIndexer converts input to string, so keys are strings
    severity_names = {
        "1": "Fatal", 
        "2": "Serious", 
        "3": "Slight",
        "1.0": "Fatal",
        "2.0": "Serious",
        "3.0": "Slight"
    }
    
    # 'labels' attribute contains the original values sorted by index (0, 1, 2...)
    for index, original_val in enumerate(label_indexer_model.labels):
        label_name = severity_names.get(str(original_val), "Unknown")
        print(f"Label {index} = {label_name} (Original: {original_val})")
        
    print("="*30 + "\n")

    spark.stop()

if __name__ == "__main__":
    run_accident_model()
