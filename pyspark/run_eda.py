import matplotlib
matplotlib.use('Agg') # <-- Use the 'Agg' backend for file-only rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import folium  # <--- NEW IMPORT
from folium.plugins import HeatMap  # <--- NEW IMPORT
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, desc, min, max, countDistinct, hour, to_date, trim, month

# Helper function to save plots
def save_plot(title, filename):
    """Saves the current matplotlib plot with standard settings."""
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Generated {filename}")

# Helper function for detailed pie chart labels
def make_autopct(values):
    """Creates a function to display both percentage and count on a pie chart."""
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val:n})'
    return my_autopct

def main():
    # 2. Create the SparkSession
    spark = SparkSession.builder \
        .appName("UK Accident EDA") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 3. Load the RAW CSV data
    print("--- Loading /home/hduser/project/uk_road_accidents.csv ---")
    df = spark.read.csv(
       "/data/uk_accidents/uk_road_accidents.csv",
        header=True,
        inferSchema=True, # We'll trust the inferSchema from inspect_data.py
        escape="\""
    )

    # 4. Feature Engineering (with human-readable labels)
    print("--- Performing feature engineering (adding labels) ---")
    df = df.withColumn("Accident_Date", to_date(trim(col("Date")), "dd/MM/yyyy"))
    df = df.withColumn("Hour", hour(col("Time")))
    df = df.withColumn("Month", month(col("Accident_Date")))
    
    # --- Map numbers to readable labels ---
    df = df.withColumn("Severity_Label", 
        when(col("Accident_Severity") == 1, "Fatal")
        .when(col("Accident_Severity") == 2, "Serious")
        .otherwise("Slight")
    )
    
    df = df.withColumn("Area_Label",
        when(col("Urban_or_Rural_Area") == 1, "Urban")
        .when(col("Urban_or_Rural_Area") == 2, "Rural")
        .otherwise("Other")
    )
    
    df = df.withColumn("Day_Name",
        when(col("Day_of_Week") == 1, "Sunday")
        .when(col("Day_of_Week") == 2, "Monday")
        .when(col("Day_of_Week") == 3, "Tuesday")
        .when(col("Day_of_Week") == 4, "Wednesday")
        .when(col("Day_of_Week") == 5, "Thursday")
        .when(col("Day_of_Week") == 6, "Friday")
        .otherwise("Saturday")
    )

    df.cache()
    print("--- Data loaded and prepped. Starting EDA plots. ---")

    # 5. Generate Visualizations

    # --- Plot 1: Pie Chart for Urban/Rural Area (Replaces Histogram) ---
    print("\n--- Generating Pie Chart for Urban/Rural Area ---")
    try:
        pd_area = df.groupBy("Area_Label").count().orderBy(desc("count")).toPandas()
        
        plt.figure(figsize=(12, 8))
        values = pd_area['count']
        labels = pd_area['Area_Label']
        
        # Create legend labels with counts
        legend_labels = [f'{l}: {v:n}' for l, v in zip(labels, values)]
        
        plt.pie(values, labels=labels, autopct=make_autopct(values), startangle=90,
                pctdistance=0.85, shadow=True)
        
        # Add a "donut hole" to make it look cleaner
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.legend(legend_labels, title="Area Types", loc="upper right")
        save_plot('Accident Distribution by Area Type', 'pie_urban_rural.png')
    except Exception as e:
        print(f"Failed to plot Pie Chart: {e}")

    # --- Plot 2: Bar Charts for Categorical Columns (with Labels) ---
    print("\n--- Generating Bar Charts for Categorical Columns ---")
    
    # Columns to plot as bar charts
    categorical_columns = {
        "Severity_Label": "Accident Severity",
        "Day_Name": "Day of Week",
        "Road_Type": "Road Type",
        "Speed_limit": "Speed Limit",
        "Light_Conditions": "Light Conditions",
        "Weather_Conditions": "Weather Conditions",
        "Road_Surface_Conditions": "Road Surface Conditions"
    }

    for col_name, title_name in categorical_columns.items():
        try:
            print(f"Plotting {col_name}...")
            # Run Spark job
            plot_data = df.groupBy(col_name).count().orderBy(desc("count")).toPandas()
            
            # For Day_Name, we need to sort it correctly
            if col_name == "Day_Name":
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                # Use .reindex() for safe sorting, in case a day is missing
                plot_data = plot_data.set_index(col_name).reindex(days_order).reset_index()

            # Create Plot
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x=col_name, y='count', data=plot_data, palette="viridis")
            
            # --- Add data labels (the "box" you requested) ---
            for p in ax.patches:
                if not pd.isna(p.get_height()): # Check for NaN values if a day was missing
                    ax.annotate(f'{int(p.get_height()):,d}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points',
                                fontweight='bold')
            
            # Clean up labels for string-based columns
            if str(plot_data[col_name].dtype) == 'object':
                 plt.xticks(rotation=45, ha='right')
            
            if not plot_data['count'].empty:
                ax.set_ylim(top=plot_data['count'].max() * 1.1) # Add space for labels
            save_plot(f'Distribution of Accidents by {title_name}', f'bar_{col_name}.png')
        except Exception as e:
            print(f"Failed to plot {col_name}: {e}")

    # --- Plot 3: Boxplot (Severity vs. Speed Limit) ---
    print("\n--- Generating Detailed Boxplot (Severity vs. Speed Limit) ---")
    try:
        # We'll use the new Severity_Label
        pd_sev_speed = df.select("Severity_Label", "Speed_limit") \
                         .filter(col("Speed_limit") > 0) \
                         .dropna() \
                         .toPandas()
        
        plt.figure(figsize=(12, 7))
        # A boxplot is the best plot to show median and outliers, as you requested
        sns.boxplot(x="Severity_Label", y="Speed_limit", data=pd_sev_speed, 
                    palette="Set1", order=["Slight", "Serious", "Fatal"])
        
        save_plot('Accident Severity by Speed Limit', 'eda_boxplot_severity_speed.png')
    except Exception as e:
        print(f"Failed to generate Boxplot: {e}")

    # --- Plot 4: Folium Heatmap (Replaces KDE) ---
    print("\n--- Generating Interactive Accident Heatmap ---")
    try:
        # Sample 50,000 points, focusing on 'Fatal' and 'Serious' accidents
        heatmap_data = df.filter(col('Accident_Severity').isin([1, 2])) \
                         .select("Latitude", "Longitude") \
                         .dropna() \
                         .sample(False, 0.1) \
                         .limit(50000) \
                         .toPandas()
        
        # Get the center of the map
        map_center = [heatmap_data['Latitude'].mean(), heatmap_data['Longitude'].mean()]
        
        # Create the map
        m = folium.Map(location=map_center, zoom_start=6)
        
        # Create the HeatMap layer
        heat_data = [[row['Latitude'], row['Longitude']] for index, row in heatmap_data.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Save to an HTML file
        m.save('eda_hotspot_map.html')
        print("Generated eda_hotspot_map.html (Open this in a browser)")
        
    except Exception as e:
        print(f"Failed to generate Folium map: {e}")

    # --- Plot 5: Correlation Heatmap (Numeric Columns) ---
    print("\n--- Generating Correlation Heatmap ---")
    try:
        # We can use the original numeric columns for correlation
        numeric_df = df.select(
            "Accident_Severity", "Day_of_Week", "Road_Type", "Light_Conditions", 
        "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area",
        "1st_Road_Class", "Junction_Control", "Pedestrian_Crossing-Physical_Facilities",
        "Special_Conditions_at_Site", "Carriageway_Hazards", "Police_Force","Longitude", "Latitude", "Number_of_Vehicles", "Speed_limit", 
        "Hour", "Month", "Year"
        ).dropna()
        
        pd_corr = numeric_df.sample(False, 0.1).limit(50000).toPandas()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pd_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        save_plot('Correlation Matrix of Numeric Features', 'eda_correlation_heatmap.png')
    except Exception as e:
        print(f"Failed to generate Correlation heatmap: {e}")

    # --- Job Done ---
    spark.stop()
    print("\n--- ✅ EDA Script Complete. All plots saved to .png files and .html file. ---")

if __name__ == "__main__":
    main()
