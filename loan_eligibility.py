import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
df = pd.read_csv('loan_dataset.csv')

# First look at the data
print(df.head())       # first 5 rows
print(df.shape)       # (rows, columns)
print(df.info())       # column types + missing count
print(df.describe())  # stats for numeric columns
# Separate categorical and numeric columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include='number').columns.tolist()

print("Categorical:", cat_cols)
print("Numeric:", num_cols)

# Check unique values per categorical column
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
    # Count how many approved vs not approved
print(df['Loan_Status'].value_counts())

# See the percentage split
print(df['Loan_Status'].value_counts(normalize=True) * 100)

# Visualise with a pie chart
counts = df['Loan_Status'].value_counts()
plt.pie(counts, labels=['Approved','Not Approved'],
        colors=['#34d399','#f472b6'], autopct='%1.1f%%')
plt.title('Loan Status Distribution')
plt.savefig('loan_target.png', dpi=150)
plt.show()
# Step 1: Count missing values per column
missing = df.isnull().sum()
print(missing[missing > 0])  # only show columns with NaN

# Step 2: Fill categorical with MODE (most frequent value)
cat_cols_with_nan = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cat_cols_with_nan:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 3: Fill numeric with MEDIAN
num_cols_with_nan = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols_with_nan:
    df[col] = df[col].fillna(df[col].median())

# Verify no more missing values
print("Remaining nulls:", df.isnull().sum().sum())
# Approval rate by Credit_History
approval_by_credit = df.groupby('Credit_History')['Loan_Status'].apply(
    lambda x: (x == 'Y').mean() * 100
)
print(approval_by_credit)

# Approval rate by Education
approval_by_edu = df.groupby('Education')['Loan_Status'].apply(
    lambda x: (x == 'Y').mean() * 100
)
print(approval_by_edu)
