import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset as a DataFrame
data = fetch_california_housing(as_frame=True)
df   = data.frame.copy()

print(df.shape)       # (20640, 9)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())  # check for missing values
# Define features and target
feats  = ['MedInc','HouseAge','AveRooms','AveBedrms',
          'Population','AveOccup','Latitude','Longitude']
target = 'MedHouseVal'

# Remove outliers: cap house prices at $500k (5.0 in $100k)
df = df[df[target] < 5.0].reset_index(drop=True)

 # Check correlation of each feature with target
corr_with_target = df[feats].corrwith(df[target])
print(corr_with_target.sort_values(ascending=False))

# Split into X (features) and y (target)
X = df[feats]      # input matrix  → shape (19648, 8)
y = df[target]     # output vector → shape (19648,)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
from sklearn.preprocessing import StandardScaler

scaler   = StandardScaler()   # create the scaler

# fit_transform: learns the mean/std of X, then scales it
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for readability
X_scaled = pd.DataFrame(X_scaled, columns=feats)

# Verify: all means ≈ 0, all stds ≈ 1
print(X_scaled.mean().round(3))
print(X_scaled.std().round(3))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,        # normalised features
    y,               # target prices
    test_size=0.2,   # 20% goes to test set
    random_state=42  # same split every time you run
)

print(f"Train: {X_train.shape}")  # (15718, 8)
print(f"Test:  {X_test.shape}")   # (3930, 8)
from sklearn.linear_model import LinearRegression
model = LinearRegression()  # create the model
model.fit(X_train, y_train) # train the model on the training data
# Predict on the test set
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# See the learned weights (coefficients)
for feat, coef in zip(feats, model.coef_):
    print(f"{feat}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)    # lower is better
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)

print(f"R²   : {r2:.4f}")    # 0.5689
print(f"RMSE : {rmse:.4f}")   # 0.6429
print(f"MAE  : {mae:.4f}")    # 0.4828

# Plot true vs predicted values
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=12)

# Draw the "perfect prediction" diagonal line
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', label='Perfect prediction')

ax.set_xlabel('Actual Price ($100k)')
ax.set_ylabel('Predicted Price ($100k)')
ax.set_title(f'Actual vs Predicted | R² = {r2:.3f}')
ax.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()




