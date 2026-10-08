import pandas as pd
from data_loader import load_data, get_column_types
from analysis_engine import run_full_analysis
from ai_insights import generate_insights

def main():
    print("Loading data...")
    # Using a dummy dataset or trying to load whatever the user might have
    # Since we don't have the user's uploaded file, let's create a small dummy df
    data = {
        'Address': ['123 Main St', '456 Oak Ave', '789 Pine Ln', '101 Maple Dr', '202 Elm St'],
        'Data Integrity': [95, 88, 92, 100, 85],
        'Suppression Constraints': ['Low', 'High', 'Medium', 'Low', 'High'],
        'Revenue': [1000, 2000, 1500, 3000, 2500]
    }
    df = pd.DataFrame(data)
    
    print("Running analysis...")
    column_types = get_column_types(df)
    analysis = run_full_analysis(df, column_types)
    
    print("Generating insights...")
    insights, err = generate_insights(
        analysis["summary"],
        analysis["statistics"],
        analysis["distribution"]
    )
    
    with open("test_out.txt", "w", encoding="utf-8") as f:
        if err:
            f.write("ERROR: " + err + "\n")
        else:
            f.write("\n=== FINAL INSIGHTS ===\n")
            f.write(insights + "\n")

if __name__ == "__main__":
    main()
