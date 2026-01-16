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
