

def get_diagnostics(df):
    shape = df.shape
    missing_values = df.isna().sum()
    dtypes = df.dtypes
    description = df.describe()
    unique_counts = df.nunique()
    
    diagnostics = {
        "Shape": shape,
        "Missing Values": missing_values,
        "Data Types": dtypes,
        "Description": description,
        "Unique Counts": unique_counts
    }
    
    cleanup_needed = True if missing_values.sum() > 0 else False
    
    return diagnostics, cleanup_needed
    