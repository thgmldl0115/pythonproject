import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset
import cx_Oracle
# Oracle 데이터베이스 연결
conn = cx_Oracle.connect("team1", "team1", "192.168.0.42:1521/xe")

# SQL 쿼리
query = """
SELECT mjr_prps_cd,  -- 주요용도코드  
       BLDG_ARCH_AREA, -- 건물건축면적
       STNDD_YR,    -- 기준년도 
       USE_MM,      -- 사용월
       ROUND(AVG(ELRW_USQNT)) as ELRW_USQNT   -- 전력 에너지 사용량
FROM ML_DATA
WHERE STNDD_YR between 2015 and 2021
GROUP BY mjr_prps_cd, STNDD_YR, USE_MM, BLDG_ARCH_AREA
ORDER BY mjr_prps_cd, STNDD_YR, TO_NUMBER(USE_MM)
"""

# 데이터 가져오기
df = pd.read_sql(query, conn)

# 연결 종료
conn.close()


# 데이터 전처리
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

df['MJR_PRPS_CD_LABEL'] = label_encoder.fit_transform(df['MJR_PRPS_CD'])
df[['ELRW_USQNT', 'BLDG_ARCH_AREA']] = scaler.fit_transform(df[['ELRW_USQNT', 'BLDG_ARCH_AREA']])

# 특성과 타겟 준비
sequence_length = 3  # 최근 3개월 데이터를 입력으로 사용
features = []
targets = []

for i in range(len(df) - sequence_length):
    sequence = df.iloc[i:i + sequence_length][['ELRW_USQNT', 'BLDG_ARCH_AREA']].values
    target = df['ELRW_USQNT'].iloc[i + sequence_length]
    features.append(sequence)
    targets.append(target)

X = np.array(features)
y = np.array(targets)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Torch Dataset 클래스 정의
class EnergyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EnergyDataset(X_train, y_train)
test_dataset = EnergyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Transformer 모델 정의
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=128, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # 시퀀스 평균을 통해 예측값 생성
        return self.fc(x)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerRegressor(input_dim=2, embed_dim=32, num_heads=4, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 50
for epoch in range(epochs):
    print("========== start ===========")
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

# 평가
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        predictions.extend(output.squeeze().cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

# 성능 평가
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print("\nTest Set Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
