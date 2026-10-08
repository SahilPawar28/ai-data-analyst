import pandas as pd
from query_engine import execute_query

def main():
    print("Creating dummy DataFrame...")
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'Sales': [1500, 25, 75, 300],
        'Region': ['North', 'South', 'East', 'West']
    }
    df = pd.DataFrame(data)
    
    question = "What is the total sales amount?"
    print(f"Asking: '{question}'")
    
    answer, code = execute_query(question, df)
    
    print("\n=== RESULT ===")
    print("Answer:", answer)
    print("Generated Code:", code)

if __name__ == "__main__":
    main()
