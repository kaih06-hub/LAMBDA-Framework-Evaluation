import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def generate_csv():
    # 1. Load the dataset
    print("📥 Fetching California Housing data...")
    data = fetch_california_housing()
    
    # 2. Create DataFrame with features
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # 3. Add the actual price (target) for reference
    # Prices are in units of $100,000
    prices = data.target
    
    # 4. Create the "Affordability" column
    # We use the median as the threshold: 
    # If price <= median, it's 'Affordable'. Otherwise, 'Not Affordable'.
    median_price = np.median(prices)
    
    # Creating a human-readable string column
    df['Affordability_Status'] = np.where(prices <= median_price, 'Affordable', 'Not Affordable')
    
    # Creating a numeric binary column (useful for ML models)
    # 1 = Affordable, 0 = Not Affordable
    df['Is_Affordable'] = (prices <= median_price).astype(int)
    
    # 5. Save to CSV
    output_filename = 'california_housing_affordability.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"✅ Success! File saved as: {output_filename}")
    print(f"Total Rows: {len(df)}")
    print(f"Median Price Threshold: ${median_price * 100_000:,.2f}")
    print("\nColumn Preview:")
    print(df[['MedInc', 'Affordability_Status', 'Is_Affordable']].head())

if __name__ == "__main__":
    generate_csv()