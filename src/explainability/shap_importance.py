# 앙상블 예측 확률
y_pred_prob = final_model.predict_proba(x_valid)[:, 1]

# 임계값 (기본 0.5)
threshold = 0.5
y_pred_label = (y_pred_prob > threshold).astype(int)

# False Positive / False Negative 인덱스
fp_idx = np.where((y_valid == 0) & (y_pred_label == 1))[0]
fn_idx = np.where((y_valid == 1) & (y_pred_label == 0))[0]
len(fp_idx), len(fn_idx)


explainer = shap.TreeExplainer(model_4)
shap_values = explainer.shap_values(x_valid)

fp_sample = fp_idx[0]


# FP / FN SHAP 평균 절대값
fp_shap_mean = np.abs(shap_values[fp_idx]).mean(axis=0)
fn_shap_mean = np.abs(shap_values[fn_idx]).mean(axis=0)

comparison_df = pd.DataFrame({
    'Feature': x_valid.columns,
    'FP_SHAP': fp_shap_mean,
    'FN_SHAP': fn_shap_mean
}).sort_values('FN_SHAP', ascending=False)

comparison_df

### 시각화
## FP_SHAP ↑ → 정상 거래에서도 과민 반응하는 변수
## FN_SHAP ↑ → 사기인데도 신호가 약한 변수


# 상위 N개 feature만 시각화
TOP_N = 10
plot_df = comparison_df.head(TOP_N)

x = np.arange(len(plot_df))
width = 0.35

plt.figure(figsize=(10, 6))

# SHAP 스타일 색상
fp_color = '#FF0051'   # SHAP red
fn_color = '#008BFB'   # SHAP blue

plt.barh(
    x - width/2,
    plot_df['FP_SHAP'],
    height=width,
    label='False Positive',
    color=fp_color
)

plt.barh(
    x + width/2,
    plot_df['FN_SHAP'],
    height=width,
    label='False Negative',
    color=fn_color
)

plt.yticks(x, plot_df['Feature'])
plt.xlabel('Mean |SHAP value|')
plt.title('FP vs FN SHAP Importance Comparison')
plt.legend()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
