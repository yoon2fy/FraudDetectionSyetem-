from sklearn.cross_decomposition import PLSRegression
import numpy as np

# === Partial Least Squares (PLS) Regression ==================================================== #
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

## === VIP Scores로 변수 선정지표 평가 =============================================== #
vip_scores = calculate_vip(pls, X_train)

vip_df = pd.DataFrame({
    'Variable': X_train.columns,
    'VIP': vip_scores
}).sort_values(by='VIP', ascending=False)

vip_df.head(10)

## === 시각화 코드 =============================================== #
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
