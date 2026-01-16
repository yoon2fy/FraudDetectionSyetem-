# Tree 기반 모델용 explainer
explainer = shap.TreeExplainer(model_4)

# 검증 데이터 기준 SHAP 값 계산
shap_values = explainer.shap_values(x_valid)

# GloBal (1)
shap.summary_plot(
    shap_values,
    x_valid,
    plot_type="dot"
)

# GloBal (2)
shap.summary_plot(
    shap_values,
    x_valid,
    plot_type="bar"
)
