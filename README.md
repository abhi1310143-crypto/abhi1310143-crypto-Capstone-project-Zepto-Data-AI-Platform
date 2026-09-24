# /analytics — Titanic Analytics Pipeline

This module loads the Titanic dataset once, profiles and cleans it, tells a
visual data story, and then builds and evaluates a full predictive-modeling
pipeline on the same cleaned data.

## Files

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Loads `sns.load_dataset('titanic')` (the only load of the raw data), profiles it, cleans it, and produces the EDA story. Saves `titanic.csv` twice: once as the raw offline snapshot right after loading, and again as the final cleaned handoff at the end. |
| `02_modeling.ipynb` | Reads the cleaned `titanic.csv` produced by `01_eda.ipynb` (no second `sns.load_dataset` call) and runs the full modeling pipeline. |
| `titanic.csv` | Committed offline fallback — the cleaned dataset, 889 rows × 14 columns, no missing values. |
| `best_titanic_pipeline.joblib` | The saved, fitted end-to-end pipeline (preprocessing + best classifier). |
| `requirements.txt` | Python dependencies. |

## Part A — Written Interpretations

**Missing-value strategy (threshold rule applied):**

| Column | % Missing | Decision |
|---|---|---|
| `deck` | 77.22% | Very high missing rate → column dropped; imputing three-quarters of a column would be unreliable and would introduce a large amount of artificial data. |
| `age` | 19.87% | Falls in the 5%–30% band → median imputed; the median is less affected by extreme ages than the mean. |
| `embarked` | 0.22% | Below 5% → affected rows dropped; the missing rate is negligible, so row removal causes very little information loss. |
| `embark_town` | 0.22% | Below 5% → affected rows dropped; these missing values occur on the same records as `embarked`, so the same row removal resolves both columns. |

Cleaned shape after applying the rules: 889 rows × 14 columns, 0 missing values.

**Univariate — age and fare:**
IQR-rule outliers: age — 65, fare — 114.
Fare statistics: mean 32.10, median 14.45, mode 8.05. Since mean > median > mode, fare is **right-skewed** — a small number of high-fare passengers create a long right tail that pulls the mean above the median and mode.

**Bivariate — survival rate (boolean masking):**
- By sex: female ≈74.2%, male ≈18.9% — a large gap.
- By class: survival rate decreases from 1st class to 3rd class.
- By sex and class combined: computed for all six sex/class groups, and the pattern holds within both sexes — being female and/or being in a higher class both independently raise survival rate.

**Correlation matrix (survived, pclass, age, sibsp, parch, fare):**
The two strongest pairwise correlations are **pclass ↔ fare (r = −0.55, negative)** and **sibsp ↔ parch (r = 0.41, positive)**. The pclass–fare relationship reflects that lower class numbers (1st class) paid higher fares; the sibsp–parch relationship reflects that passengers traveling with more siblings/spouses also tended to travel with more parents/children, i.e., larger families. These are measured linear associations in the cleaned data and do not establish causation.

**Multivariate data story (4 charts):**
1. *Survival rate by sex* — women survived at a much higher rate than men, consistent with a "women first" evacuation pattern.
2. *Survival rate by class* — survival declines from 1st to 3rd class, consistent with the negative pclass–survived correlation.
3. *Age distribution by survival* — survivors and non-survivors' age distributions overlap substantially; survivors have a somewhat lower median age, but age alone does not clearly separate the two groups.
4. *Fare by class and survival* — within each class, survivors generally have higher fare distributions than non-survivors, suggesting fare (a proxy for accommodation/deck location) carried additional survival information beyond class alone.

**EDA-stage standardization check:** age and fare were z-scored with `StandardScaler`; the printed before/after means and standard deviations confirm the scaled columns have approximately mean 0 and standard deviation 1. This check is exploratory only and does not feed into `titanic.csv` or the modeling pipeline.

## Part B — Model Comparison Table

**Classifiers (test set, identical stratified 80/20 split):**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 |
| Random Forest | 0.809 | 0.766 | 0.721 | 0.742 | 0.820 |

**Imbalance-handling comparison (Logistic Regression, SMOTE applied to the training fold only):**

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.783 | 0.691 | 0.734 |
| `class_weight='balanced'` | 0.718 | 0.750 | 0.734 |
| SMOTE | 0.735 | 0.735 | 0.735 |

SMOTE produced the most even precision/recall trade-off (best F1 of the three) on this test set.

**Random Forest tuning (`GridSearchCV`, `oob_score=True`):**
Best parameters: `max_depth=5, max_features='sqrt', n_estimators=200`. Best CV accuracy: 0.82. OOB score: 0.8214. Tuned RF test performance: accuracy 0.8315, F1 0.75.

**Regression side-task (predicting fare — separate metric group, not comparable to classification metrics):**

| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 21.10 | 41.70 | 0.348 | 0.309 |

The residual plot and a supporting spread check (residual standard deviation ratio ≈4.33 between the low- and high-predicted-fare halves of the test set) both indicate **heteroscedasticity** — the residual variance is not constant and grows as predicted fare increases.

## Final Recommendation

Based on the test-set results, **Random Forest** is the classifier recommended for deployment. It ties Logistic Regression on accuracy (0.809) and trails it slightly on AUC (0.820 vs 0.861), but it has the highest F1 score (0.742) of the three models, driven by a better recall (0.721 vs 0.691 for Logistic Regression and 0.559 for the Decision Tree) without a large drop in precision (0.766). For a survival-prediction task, correctly identifying more true survivors (recall) while keeping false positives reasonably low (precision) is the more relevant trade-off than accuracy or AUC alone, which is why Random Forest is preferred over Logistic Regression and the Decision Tree here.

The complete fitted pipeline (preprocessing + tuned Random Forest) was saved with `joblib.dump()` to `best_titanic_pipeline.joblib`. Reloading it with `joblib.load()` and predicting on raw, unprocessed test input reproduced identical predictions to the original in-memory pipeline, confirming it is usable end-to-end on new raw data.