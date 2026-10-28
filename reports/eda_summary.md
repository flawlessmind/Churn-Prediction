# EDA summary (генерируется автоматически: `python -m src.eda`)

## 1. Target
Ушли 1869 из 7043 (26.5%); дисбаланс ≈ 1:2.8. Accuracy вводит в заблуждение (константа «остался» даёт 73.5%) → основная метрика PR-AUC, StratifiedKFold.

## 2. Пропуски
NaN нет. `TotalCharges`: 11 значений — пробелы (из-за них колонка читается как object). Все они у клиентов с tenure = 0: True → заполняем MonthlyCharges × tenure = 0.

## 3. Числовые признаки
| признак | skew | выбросов по IQR |
|---|---|---|
| tenure | 0.24 | 0 |
| MonthlyCharges | -0.22 | 0 |
| TotalCharges | 0.96 | 0 |

Медиана tenure: ушедшие 10 мес. vs оставшиеся 38 мес. Выбросы в смысле ошибок данных отсутствуют; TotalCharges скошен вправо → верхний хвост режется (winsorizing), для деревьев распределение не трансформируется.

## 4. Churn rate по категориям (группы n ≥ 100)
**Самые рискованные:**
- PaymentMethod = Electronic check: 45.3% (n=2365)
- Contract = Month-to-month: 42.7% (n=3875)
- InternetService = Fiber optic: 41.9% (n=3096)
- OnlineSecurity = No: 41.8% (n=3498)
- SeniorCitizen = 1.0: 41.7% (n=1142)

**Самые надёжные:**
- Contract = Two year: 2.8% (n=1695)
- InternetService = No: 7.4% (n=1526)
- DeviceProtection = No internet service: 7.4% (n=1526)
- OnlineSecurity = No internet service: 7.4% (n=1526)
- TechSupport = No internet service: 7.4% (n=1526)

## 5. Корреляции
**Сильнее всего связаны с Churn:**
- Contract_Month_to_month: +0.41
- tenure: -0.35
- is_new_customer: +0.31
- InternetService_Fiber_optic: +0.31
- Contract_Two_year: -0.30
- PaymentMethod_Electronic_check: +0.30
- InternetService_No: -0.23
- has_autopay: -0.21

**Мультиколлинеарные пары (|r| > 0.8):**
- tenure ~ TotalCharges: +0.83
- MonthlyCharges ~ avg_charge_per_month: +0.96

Линейной модели мультиколлинеарность вредит (поэтому L2-регуляризация + StandardScaler); для бустинга/леса — не проблема.

## Вывод
- Информативны: тип контракта, tenure, тип интернета, способ оплаты, наличие техподдержки/онлайн-безопасности.
- Слабо информативны: gender и ряд флагов (см. churn rate ≈ среднему).
- Требуют преобразования: TotalCharges (пробелы, скошенность), категории с «No internet service» (сводятся к 0), tenure (нелинейность → is_new_customer), цена/длительность (→ avg_charge_per_month).
