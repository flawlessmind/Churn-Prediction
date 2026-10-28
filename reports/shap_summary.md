# SHAP: факторы ухода

## Топ-5 факторов, ПОВЫШАЮЩИХ риск ухода
1. `Contract_Month_to_month` — mean|SHAP| = 0.736
2. `InternetService_Fiber_optic` — mean|SHAP| = 0.280
3. `PaymentMethod_Electronic_check` — mean|SHAP| = 0.172
4. `MonthlyCharges` — mean|SHAP| = 0.142
5. `PaperlessBilling` — mean|SHAP| = 0.123

## Топ-5 факторов, СНИЖАЮЩИХ риск ухода
1. `tenure` — mean|SHAP| = 0.330
2. `Contract_Two_year` — mean|SHAP| = 0.162
3. `InternetService_No` — mean|SHAP| = 0.150
4. `OnlineSecurity` — mean|SHAP| = 0.090
5. `TotalCharges` — mean|SHAP| = 0.080

Графики: `reports/figures/shap_bar.png`, `shap_beeswarm.png`, `shap_waterfall_*.png`.
