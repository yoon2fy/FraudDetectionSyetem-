## Local (1)
idx = 0  # 보고 싶은 검증 데이터 인덱스

shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    x_valid.iloc[idx],
    matplotlib=True
)

## Local (2)
shap.dependence_plot(
    'V17',
    shap_values,
    x_valid
)
