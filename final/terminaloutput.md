mework8.py                                                                                                                 ======================================================================
PHASE 1: DATA PREPARATION
======================================================================
Training data shape: (576, 15)

Training data columns:
['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'clinic', 
'hd']                                                                                                                      
First 5 rows of training data:
    age  sex   cp  trestbps   chol  fbs  restecg  thalach  exang  oldpeak  slope   ca  thal clinic  hd
0  49.0  0.0  3.0     160.0  180.0  0.0      0.0    156.0    0.0      1.0    2.0  NaN   NaN   hung   1
1  62.0  0.0  4.0     124.0  209.0  0.0      0.0    163.0    0.0      0.0    1.0  0.0   3.0   clev   0
2  60.0  1.0  3.0     115.0    0.0  NaN      0.0    143.0    0.0      2.4    1.0  NaN   NaN   swit   1
3  65.0  1.0  4.0     160.0    0.0  1.0      1.0    122.0    0.0      NaN    NaN  NaN   7.0   swit   1
4  34.0  1.0  1.0     118.0  182.0  0.0      2.0    174.0    0.0      0.0    1.0  0.0   3.0   clev   0

Data types:
age         float64
sex         float64
cp          float64
trestbps    float64
chol        float64
fbs         float64
restecg     float64
thalach     float64
exang       float64
oldpeak     float64
slope       float64
ca          float64
thal        float64
clinic       object
hd            int64
dtype: object

Basic statistics:
              age         sex          cp    trestbps  ...       slope          ca        thal          hd
count  576.000000  576.000000  576.000000  574.000000  ...  409.000000  245.000000  323.000000  576.000000
mean    51.921875    0.748264    3.180556  131.503484  ...    1.706601    0.726531    5.058824    0.500000
std      9.413356    0.434388    0.957553   18.510008  ...    0.587434    0.959588    1.928764    0.500435
min     28.000000    0.000000    1.000000   80.000000  ...    1.000000    0.000000    3.000000    0.000000
25%     45.000000    0.000000    2.000000  120.000000  ...    1.000000    0.000000    3.000000    0.000000
50%     53.000000    1.000000    4.000000  130.000000  ...    2.000000    0.000000    6.000000    0.500000
75%     58.000000    1.000000    4.000000  140.000000  ...    2.000000    1.000000    7.000000    1.000000
max     77.000000    1.000000    4.000000  200.000000  ...    3.000000    3.000000    7.000000    1.000000

[8 rows x 14 columns]

### 1.2 Document Variables ###

Continuous variables: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
Categorical variables: ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'clinic']
Target variable: hd

Categorical variable levels and frequencies:

  sex:
    0.0: 145 (25.2%)
    1.0: 431 (74.8%)

  cp:
    1.0: 32 (5.6%)
    2.0: 125 (21.7%)
    3.0: 126 (21.9%)
    4.0: 293 (50.9%)

  fbs:
    0.0: 454 (78.8%)
    1.0: 57 (9.9%)
    Missing: 65 (11.3%)

  restecg:
    0.0: 366 (63.5%)
    1.0: 75 (13.0%)
    2.0: 134 (23.3%)
    Missing: 1 (0.2%)

  exang:
    0.0: 382 (66.3%)
    1.0: 192 (33.3%)
    Missing: 2 (0.3%)

  slope:
    1.0: 148 (25.7%)
    2.0: 233 (40.5%)
    3.0: 28 (4.9%)
    Missing: 167 (29.0%)

  ca:
    0.0: 138 (24.0%)
    1.0: 53 (9.2%)
    2.0: 37 (6.4%)
    3.0: 17 (3.0%)
    Missing: 331 (57.5%)

  thal:
    3.0: 149 (25.9%)
    6.0: 31 (5.4%)
    7.0: 143 (24.8%)
    Missing: 253 (43.9%)

  clinic:
    clev: 241 (41.8%)
    hung: 235 (40.8%)
    swit: 100 (17.4%)

Target distribution (hd):
hd
1    288
0    288
Name: count, dtype: int64
Proportion with heart disease: 0.500

### 1.3 Handle Missing Values ###

Missing values per column:
age           0
sex           0
cp            0
trestbps      2
chol         19
fbs          65
restecg       1
thalach       2
exang         2
oldpeak       5
slope       167
ca          331
thal        253
clinic        0
hd            0
dtype: int64

Checking for zeros that should be treated as missing:
  trestbps == 0: 0 rows
  chol == 0: 100 rows

After replacing 0s with NaN:
Missing values per column (training):
age           0
sex           0
cp            0
trestbps      2
chol        119
fbs          65
restecg       1
thalach       2
exang         2
oldpeak       5
slope       167
ca          331
thal        253
clinic        0
hd            0
dtype: int64

Missingness by clinic:

hung (n=235):
  trestbps: 0.4% missing
  chol: 8.1% missing
  fbs: 2.1% missing
  thalach: 0.4% missing
  exang: 0.4% missing
  slope: 65.1% missing
  ca: 98.7% missing
  thal: 88.9% missing

clev (n=241):
  ca: 1.7% missing
  thal: 0.8% missing

swit (n=100):
  trestbps: 1.0% missing
  chol: 100.0% missing
  fbs: 60.0% missing
  restecg: 1.0% missing
  thalach: 1.0% missing
  exang: 1.0% missing
  oldpeak: 5.0% missing
  slope: 14.0% missing
  ca: 95.0% missing
  thal: 42.0% missing

### 1.4 Handle Outliers ###

Checking for outliers in continuous variables:
  age: 0 outliers (range: 28.0 - 77.0)
  trestbps: 16 outliers (range: 80.0 - 200.0)
  chol: 12 outliers (range: 85.0 - 603.0)
  thalach: 1 outliers (range: 67.0 - 195.0)
  oldpeak: 9 outliers (range: -2.6 - 6.2)

### Summary Statistics for Report ###

Continuous Variables Summary:
          count        mean        std   min    max  missing  missing_pct
age       576.0   51.921875   9.413356  28.0   77.0        0          0.0
trestbps  574.0  131.503484  18.510008  80.0  200.0        2          0.3
chol      457.0  246.547046  58.125304  85.0  603.0      119         20.7
thalach   574.0  141.222997  25.220682  67.0  195.0        2          0.3
oldpeak   571.0    0.785289   1.064232  -2.6    6.2        5          0.9

======================================================================
======================================================================

======================================================================
PHASE 2: FEATURE ENGINEERING
======================================================================

### 2.1 Encode Categorical Variables ###

Converting numeric categoricals to string type for encoding...
Categorical variables after conversion:
  sex: ['0', '1']
  cp: ['1', '2', '3', '4']
  fbs: ['0', '1']
  restecg: ['0', '1', '2']
  exang: ['0', '1']
  slope: ['1', '2', '3']
  ca: ['0', '1', '2', '3']
  thal: ['3', '6', '7']
  clinic: ['clev', 'hung', 'swit']

### 2.2 Check Distributions of Continuous Variables ###

Skewness of continuous variables:
  age: -0.140
  trestbps: 0.696
  chol: 1.562
  thalach: -0.346
  oldpeak: 1.207

No log transforms needed - skewness is moderate and
standardization will handle scale differences.

### 2.5 Define Preprocessing Pipeline ###

X_train shape: (576, 14)
y_train shape: (576,)

Continuous variables: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
Categorical variables: ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'ca', 'thal', 'clinic']

Preprocessing pipeline defined:
  - Continuous: median imputation + StandardScaler
  - Categorical: mode imputation + OneHotEncoder (drop first)

Total features after preprocessing: 22
Feature names:
  1. age
  2. trestbps
  3. chol
  4. thalach
  5. oldpeak
  6. sex_1
  7. fbs_1
  8. exang_1
  9. cp_2
  10. cp_3
  11. cp_4
  12. restecg_1
  13. restecg_2
  14. slope_2
  15. slope_3
  16. ca_1
  17. ca_2
  18. ca_3
  19. thal_6
  20. thal_7
  21. clinic_hung
  22. clinic_swit

Transformed X_train shape: (576, 22)

======================================================================
======================================================================

======================================================================
PHASE 3: MODEL FITTING & SELECTION
======================================================================

### 3.1 Validation Strategy ###

Dev (train-split) size: 460 rows
Validation size: 116 rows

### 3.2 Candidate Models ###

Candidate models:
  - Full logistic (all features)
  - Forward selection (10 features)
  - Forward selection (15 features)

Penalization: none (no Ridge/Lasso).

### 3.3 Model Comparison ###

Model comparison (selection uses validation AUC, tie-break accuracy):
                          model cv_auc_mean cv_auc_sd cv_acc_mean cv_acc_sd val_auc val_acc
Forward selection (10 features)       0.895     0.034       0.826     0.034   0.929   0.871
Forward selection (15 features)       0.907     0.029       0.830     0.048   0.927   0.862
   Full logistic (all features)       0.905     0.036       0.837     0.034   0.926   0.888

Selected model: Forward selection (10 features)

======================================================================
======================================================================

======================================================================
PHASE 4: FINAL MODEL EVALUATION
======================================================================

### 4.1 Refit on Full Training Data ###


### 4.2 Performance Metrics (Training + 5-fold CV) ###

Performance table:
  metric train cv_mean cv_sd
     AUC 0.931   0.906 0.039
Accuracy 0.856   0.825 0.060

======================================================================
======================================================================

======================================================================
PHASE 5: MODEL INTERPRETATION
======================================================================

### 5.1 Statsmodels Refit for Odds Ratios + CI ###

Top coefficients by absolute value (excluding intercept):
    feature  coef odds_ratio ci_low ci_high
clinic_swit 3.725     41.456 15.809 108.713
       ca_1 2.202      9.042  3.386  24.143
       ca_2 1.972      7.187  1.745  29.594
       cp_4 1.903      6.703  3.887  11.560
    slope_2 1.585      4.879  2.541   9.368
     thal_7 1.278      3.590  1.857   6.939
      sex_1 1.041      2.831  1.493   5.370
    exang_1 1.035      2.816  1.546   5.129
      fbs_1 0.877      2.404  0.966   5.982
    oldpeak 0.713      2.039  1.508   2.758

======================================================================
======================================================================

======================================================================
PHASE 6: VISUALIZATION
======================================================================

### 6.1 ROC Curve (Training) ###

Saved ROC figure to: C:\github\aihc\aihc5615\week11HW8\roc_curve.png

======================================================================
======================================================================
                           Logit Regression Results
==============================================================================
Dep. Variable:                      y   No. Observations:                  576
Model:                          Logit   Df Residuals:                      565
Method:                           MLE   Df Model:                           10
Date:                Tue, 06 Jan 2026   Pseudo R-squ.:                  0.5236
Time:                        23:59:47   Log-Likelihood:                -190.22
converged:                       True   LL-Null:                       -399.25
Covariance Type:            nonrobust   LLR p-value:                 1.337e-83
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
const          -4.3314      0.474     -9.130      0.000      -5.261      -3.401
oldpeak         0.7125      0.154      4.625      0.000       0.411       1.014
sex_1           1.0407      0.327      3.186      0.001       0.401       1.681
fbs_1           0.8772      0.465      1.886      0.059      -0.034       1.789
exang_1         1.0353      0.306      3.385      0.001       0.436       1.635
cp_4            1.9026      0.278      6.843      0.000       1.358       2.448
slope_2         1.5849      0.333      4.761      0.000       0.932       2.237
ca_1            2.2019      0.501      4.394      0.000       1.220       3.184
ca_2            1.9723      0.722      2.731      0.006       0.557       3.388
thal_7          1.2782      0.336      3.801      0.000       0.619       1.937
clinic_swit     3.7246      0.492      7.572      0.000       2.761       4.689
===============================================================================
