# 26/01/11 upload
## imports
import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import lightgbm as lgb
from matplotlib import cm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import mutual_info_classif

## 파일 다운로드
google_path = 'https://drive.google.com/uc?id='
file_id = '1cA2bkyBdPvNFX8yiL-kyczqlfv8YvjLK'
output_name = 'train.csv'
gdown.download(google_path+file_id, output_name)
df = pd.read_csv('train.csv')
df.head()

# 데이터 분석 ---------------------------------------------------------------------------- #
## Missing Data Check
df.isnull().sum()
df.isnull().sum()[df.isnull().sum() > 0] # 결측치가 하나라도 있는 컬럼 확인

## Data Scaling
### Feature & Target Split
X = df.drop('Class', axis=1)
y = df['Class']

### Log transform
#### V1 ~ V28에 음수가 많음. 따라서 Amount, Time만 Log 변환하였음. 
X_log = X.copy()
log_cols = ['Amount', 'Time']
for col in log_cols:
    X_log[col] = np.log(X_log[col] + 1)

### Min-Max Scaling
X_scaled = X_log.copy()
for col in X_scaled.columns:
    col_min = X_scaled[col].min()
    col_max = X_scaled[col].max()
    X_scaled[col] = (X_scaled[col] - col_min) / (col_max - col_min)
X_scaled.describe()

# 변수 선정 ---------------------------------------------------------------------------- #
## Partial Least Squares (PLS) Regression
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)

def calculate_vip(pls, X):
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    p, h = w.shape
    vip = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([
            (w[i, j] ** 2) * s[j]
            for j in range(h)
        ])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)

    return vip

### VIP Scores로 변수 선정지표 평가
vip_scores = calculate_vip(pls, X_train)

vip_df = pd.DataFrame({
    'Variable': X_train.columns,
    'VIP': vip_scores
}).sort_values(by='VIP', ascending=False)

vip_df.head(10)

### 시각화 코드 --------------------------------------------------------- #
important_vars = vip_df[vip_df['VIP'] >= 1.21]
important_vars

pls = PLSRegression(n_components=3)
pls.fit(X_scaled, y)

# Scores (샘플 좌표)
T = pls.x_scores_        # (n_samples, n_components)

# Loadings (변수 방향)
P = pls.x_loadings_      # (n_features, n_components)

feature_names = X_scaled.columns
plt.figure(figsize=(8, 8))

# ----- scores (점) -----
sc = plt.scatter(
    T[:, 0], T[:, 1],
    c=y,               
    cmap='jet',
    alpha=0.6,
    s=10
)

plt.colorbar(sc, label='y축')

# ----- loadings (화살표) -----
scale = 200  # 화살표 길이 조절 (논문 느낌 핵심)

for i, var in enumerate(feature_names):
    plt.arrow(
        0, 0,
        P[i, 0] * scale,
        P[i, 1] * scale,
        color='blue',
        alpha=0.7,
        head_width=5
    )
    plt.text(
        P[i, 0] * scale * 1.1,
        P[i, 1] * scale * 1.1,
        var,
        fontsize=9
    )

# 축선
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PLS Biplot (2 Components)')
plt.grid(True)
plt.show()



fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

# ----- scores -----
sc = ax.scatter(
    T[:, 0], T[:, 1], T[:, 2],
    c=y,
    cmap='jet',
    alpha=0.6,
    s=10
)

fig.colorbar(sc, ax=ax, label='y축')

# ----- loadings (3D arrows) -----
scale = 200

for i, var in enumerate(feature_names):
    ax.quiver(
        0, 0, 0,
        P[i, 0] * scale,
        P[i, 1] * scale,
        P[i, 2] * scale,
        color='blue',
        arrow_length_ratio=0.05
    )
    ax.text(
        P[i, 0] * scale * 1.1,
        P[i, 1] * scale * 1.1,
        P[i, 2] * scale * 1.1,
        var,
        fontsize=8
    )

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('PLS Biplot (3 Components)')
plt.show()

vip_threshold = 1.21

vip_selected = vip_df[vip_df['VIP'] >= vip_threshold] \
                    .sort_values(by='VIP', ascending=False)

vip_selected

plt.figure(figsize=(10, 6))

bars = plt.bar(
    vip_selected['Variable'],
    vip_selected['VIP'],
    color='tab:blue'
)

# 기준선
plt.axhline(
    y=vip_threshold,
    color='red',
    linestyle='--',
    linewidth=1
)

plt.text(
    -0.5,
    vip_threshold + 0.05,
    'VIP ≥ 1.21',
    fontsize=11
)

plt.ylabel('VIP Scores')
plt.xlabel('Variables')
plt.title('VIP Scores from PLS Regression')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

bars = plt.bar(
    vip_df['Variable'],
    vip_df['VIP'],
    color='tab:blue'
)

# 기준선
plt.axhline(
    y=vip_threshold,
    color='red',
    linestyle='--',
    linewidth=1
)

plt.text(
    -0.5,
    vip_threshold + 0.05,
    'VIP ≥ 1.21',
    fontsize=11
)

plt.ylabel('VIP Scores')
plt.xlabel('Variables')
plt.title('VIP Scores from PLS Regression')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


### 변수 선정을 위해 통합된 스코어를 계산합니다.
# 1-1. Pearson Correlation
pearson_scores = {}
for col in X_scaled.columns:
    pearson_scores[col] = abs(pearsonr(X_scaled[col], y)[0])

# 1-2. Spearman Rank Correlation
spearman_scores = {}
for col in X_scaled.columns:
    spearman_scores[col] = abs(spearmanr(X_scaled[col], y)[0])

# 1-3. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
ridge_scores = dict(zip(X_scaled.columns, abs(ridge.coef_)))

# 1-4. Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)
lasso_scores = dict(zip(X_scaled.columns, abs(lasso.coef_)))

# 1-5. Elastic Net
enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
enet.fit(X_scaled, y)
enet_scores = dict(zip(X_scaled.columns, abs(enet.coef_)))

# 1-6. Mutual Information Score (MIS)
mi = mutual_info_classif(X_scaled, y, random_state=42)
mi_scores = dict(zip(X_scaled.columns, mi))

# 1-7. PLS – VIP Score
pls = PLSRegression(n_components=3)
pls.fit(X_scaled, y)
vip_scores = calculate_vip(pls, X_scaled)
vip_scores = dict(zip(X_scaled.columns, vip_scores))

# 2-1. 하나의 DataFrame으로 결합
fs_df = pd.DataFrame({
    'Pearson': pearson_scores,
    'Spearman': spearman_scores,
    'Ridge': ridge_scores,
    'Lasso': lasso_scores,
    'ElasticNet': enet_scores,
    'MI': mi_scores,
    'VIP': vip_scores
})

# 2-2. [0,1] 정규화 (Radar plot 필수)
fs_norm = (fs_df - fs_df.min()) / (fs_df.max() - fs_df.min())

# 3. 최종 Feature Selection (합의 기반)
# 평균 중요도 기준
fs_norm['MeanScore'] = fs_norm.mean(axis=1)

# 상위 변수 확인
fs_norm.sort_values('MeanScore', ascending=False).head(10)

# 주요 변수 선정
main_features = fs_norm.sort_values(
    'MeanScore', ascending=False
).head(10).index.tolist()

main_features # 선정된 Top 10 변수명 선정


# 4-1. 시각화 대상 변수 선택
plot_features = ['V17', 'V14', 'V12', 'V10', 'V7', 'V16', 'V3', 'V11', 'V4', 'V9'] # main_features
methods = ['Pearson', 'Spearman', 'Ridge', 'Lasso', 'ElasticNet', 'MI', 'VIP']

plot_data = fs_norm.loc[plot_features, methods]
plot_data.head(5)

# 시각화 =================================================
# 1. 기본 설정
N = len(plot_features)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

# 2. Consensus score (methods 평균)
mean_scores = plot_data.mean(axis=1).values
norm = plt.Normalize(0, 1)
colors = cm.Reds(norm(mean_scores))

# 3. 배경 섹터
ax.bar(
    angles,
    mean_scores,
    width=2*np.pi/N * 0.75,
    color=colors,
    alpha=0.65,
    edgecolor='black',
    linewidth=1.3,
    zorder=1
)

# 4. Method별 dashed line + scatter
markers = ['o', 's', '^', 'D', 'v', 'P', '*']
method_colors = cm.tab10(np.linspace(0, 1, len(methods)))

for i, method in enumerate(methods):
    values = plot_data[method].values

    # dashed line
    ax.plot(
        np.append(angles, angles[0]),
        np.append(values, values[0]),
        linestyle='--',
        linewidth=1.2,
        color=method_colors[i],
        alpha=0.75,
        zorder=3
    )

    # scatter points
    ax.scatter(
        angles,
        values,
        s=55,
        marker=markers[i % len(markers)],
        color=method_colors[i],
        edgecolors='black',
        linewidth=0.6,
        zorder=4,
        label=method
    )

# 5. Best feature (Mean 기준) 강조
best_idx = np.argmax(mean_scores)

ax.bar(
    angles[best_idx],
    mean_scores[best_idx],
    width=2*np.pi/N * 0.75,
    color='none',
    edgecolor='darkred',
    linewidth=3,
    zorder=6
)

# 6. 축 / 라벨 스타일
ax.set_xticks(angles)
ax.set_xticklabels(plot_features, fontsize=12, fontweight='bold')

ax.set_ylim(0, 1)
ax.set_yticklabels([])
ax.grid(alpha=0.3)

ax.set_title(
    "Feature Selection Results (Consensus-based Importance)",
    fontsize=16,
    pad=28
)

# 7. 컬러바 (Mean Importance)
sm = cm.ScalarMappable(norm=norm, cmap='Reds')
cbar = plt.colorbar(sm, ax=ax, pad=0.12)
cbar.set_label('Mean Feature Importance', fontsize=11)

# 8. 범례
ax.legend(
    bbox_to_anchor=(1.18, 1.05),
    fontsize=9,
    frameon=False
)
plt.tight_layout()
plt.show()

# ====================================================================
