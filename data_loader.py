import pandas as pd

def load_data(uploaded_file):
    """
    Accepts a file uploaded via Streamlit.
    Supports CSV and Excel formats.
    Returns a cleaned Pandas DataFrame.
    """
    try:
        file_name = uploaded_file.name

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)

        else:
            return None, "❌ Unsupported file format. Please upload a CSV or Excel file."

        # Basic cleaning
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # Normalize column names
        df.dropna(how="all", inplace=True)  # Drop fully empty rows

        return df, None  # (dataframe, error)

    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}"


def get_column_types(df):
    """
    Detects and categorizes columns into:
    - numeric
    - categorical
    - date
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Try to detect date columns from object columns
    date_cols = []
    for col in categorical_cols:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except:
            pass

    # Remove detected date cols from categorical
    categorical_cols = [c for c in categorical_cols if c not in date_cols]

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols
    }
