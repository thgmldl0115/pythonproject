import cx_Oracle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV


# Oracle 데이터베이스 연결
conn = cx_Oracle.connect("team1", "team1", "192.168.0.42:1521/xe")

# SQL 쿼리
query = """
SELECT a.mjr_prps_cd,  -- 주요용도코드  
       b.STNDD_YR,    -- 기준년도 
       b.USE_MM,      -- 사용월
       ROUND(AVG(b.ELRW_USQNT)) as ELRW_USQNT   -- 전력 에너지 사용량
FROM building_usage1 a, daejeon_energy1 b
WHERE a.lotno_addr = b.lotno_addr
AND a.mjr_prps_nm IS NOT NULL
AND b.stndd_yr between 2015 and 2022
GROUP BY mjr_prps_cd, STNDD_YR, USE_MM
ORDER BY mjr_prps_cd, STNDD_YR, TO_NUMBER(USE_MM)
"""

# 데이터 가져오기
df = pd.read_sql(query, conn)

# 연결 종료
conn.close()

# 데이터 준비 및 전처리
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Label encode 'MJR_PRPS_CD' and save label mapping
df['MJR_PRPS_CD_LABEL'] = label_encoder.fit_transform(df['MJR_PRPS_CD'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Normalize 'ELRW_USQNT' using Min-Max scaling
df[['ELRW_USQNT']] = scaler.fit_transform(df[['ELRW_USQNT']])

# Create a new column for the target variable (next month's usage)
df['NEXT_ELRW_USQNT'] = df['ELRW_USQNT'].shift(-1)

# Drop rows with NaN values in the target column (due to shift operation)
df = df.dropna()

# Define features (X) and target (y) using the normalized data
X = df[['MJR_PRPS_CD_LABEL', 'STNDD_YR', 'USE_MM', 'ELRW_USQNT']]
y = df['NEXT_ELRW_USQNT']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("\n최적 하이퍼파라미터:", best_params)
print("\n최적 모델 평가")
print(f"RMSE: {rmse_best}")
print(f"MAE: {mae_best}")
print(f"R² Score: {r2_best}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", marker='o', linestyle='--')
plt.plot(y_pred_best.flatten(), label="Predicted (Best Model)", marker='x', linestyle='-')
plt.legend()
plt.title("Actual vs Predicted (Test Set) - Best Model")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Usage")
plt.show()