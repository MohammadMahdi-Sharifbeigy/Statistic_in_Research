# Repeated Measures ANOVA Analysis Report
## 95% Confidence Interval Analysis

---

## 1. Data Import and Preparation

### Import Data
```stata
import delimited "D:\Cognitive sience\Summer404\Statistic in Research\Prism\DPLFC\ANOVA\repeated_anova_p005_long.csv", clear
```

**Output:**
```
(encoding automatically selected: ISO-8859-1)
(3 vars, 100 obs)
```

### Create Numeric Time Variable

```stata
encode time, gen(time_num)
gen time_n = .
replace time_n = 0 if time == "Baseline"
replace time_n = 1 if time == "Month_1"
replace time_n = 2 if time == "Month_2"
replace time_n = 3 if time == "Month_3"
```

**Output:**
```
(100 missing values generated)
(25 real changes made)
(25 real changes made)
(25 real changes made)
(25 real changes made)
```

### Label Time Variable

```stata
label define timelbl 0 "Baseline" 1 "Month 1" 2 "Month 2" 3 "Month 3"
label values time_n timelbl
```

### Verify Data Structure

```stata
summarize
```

**Output:**
```
    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
  subject_id |        100          13    7.247431          1         25
        time |          0
       value |        100     69.1868    13.93395      32.75      99.72
    time_num |        100         2.5    1.123666          1          4
      time_n |        100         1.5    1.123666          0          3
```

```stata
list in 1/10
```

**Output:**
```
     +---------------------------------------------------+
     | subjec~d       time   value   time_num     time_n |
     |---------------------------------------------------|
  1. |        1   Baseline   70.84   Baseline   Baseline |
  2. |        2   Baseline   76.49   Baseline   Baseline |
  3. |        3   Baseline   77.33   Baseline   Baseline |
  4. |        4   Baseline   70.12   Baseline   Baseline |
  5. |        5   Baseline   68.38   Baseline   Baseline |
     |---------------------------------------------------|
  6. |        6   Baseline   76.63   Baseline   Baseline |
  7. |        7   Baseline   56.43   Baseline   Baseline |
  8. |        8   Baseline   75.59   Baseline   Baseline |
  9. |        9   Baseline    65.9   Baseline   Baseline |
 10. |       10   Baseline   75.19   Baseline   Baseline |
     +---------------------------------------------------+
```

---

## 2. Repeated Measures ANOVA

### Run the Analysis (98% Confidence Level)

```stata
anova value subject_id time_n, repeated(time_n)
```

**Output:**

```
                         Number of obs =        100    R-squared     =  0.9082
                         Root MSE      =    4.95034    Adj R-squared =  0.8738

                  Source | Partial SS         df         MS        F    Prob>F
              -----------+----------------------------------------------------
                   Model |  17456.913         27   646.55232     26.38  0.0000
                         |
              subject_id |  16958.999         24   706.62497     28.83  0.0000
                  time_n |  497.91357          3   165.97119      6.77  0.0004
                         |
                Residual |  1764.4223         72   24.505865  
              -----------+----------------------------------------------------
                   Total |  19221.335         99    194.1549  


Between-subjects error term:  subject_id
                     Levels:  25        (24 df)
     Lowest b.s.e. variable:  subject_id

Repeated variable: time_n
                                          Huynh-Feldt epsilon        =  0.9805
                                          Greenhouse-Geisser epsilon =  0.8660
                                          Box's conservative epsilon =  0.3333

                                            ------------ Prob > F ------------
                  Source |     df      F    Regular    H-F      G-G      Box
              -----------+----------------------------------------------------
                  time_n |      3     6.77   0.0004   0.0005   0.0009   0.0156
                Residual |     72
              ----------------------------------------------------------------
```

### Key Findings from ANOVA Table

| Metric | Value |
|--------|-------|
| **Sample Size** | N = 100 (25 subjects × 4 time points) |
| **R-squared** | 0.9082 (90.82% variance explained) |
| **Adjusted R-squared** | 0.8738 |
| **Root MSE** | 4.95034 |

#### Main Effects

| Source | SS | df | MS | F | p-value |
|--------|----|----|----|----|---------|
| **Model** | 17456.91 | 27 | 646.55 | 26.38 | **< 0.0001*** |
| **subject_id** | 16958.99 | 24 | 706.62 | 28.83 | **< 0.0001*** |
| **time_n** | 497.91 | 3 | 165.97 | **6.77** | **0.0004*** |
| **Residual** | 1764.42 | 72 | 24.51 | | |
| **Total** | 19221.34 | 99 | 194.15 | | |

**Interpretation:**
- ✅ **Time effect is highly significant** (F(3,72) = 6.77, p = 0.0004)
- ✅ Strong between-subject variability (F = 28.83, p < 0.0001)
- ✅ The model explains 90.82% of the variance

#### Sphericity Assumption Tests

| Test | Epsilon (ε) | Adjusted p-value |
|------|-------------|------------------|
| **Regular (assumed)** | 1.000 | 0.0004 |
| **Huynh-Feldt** | 0.9805 | 0.0005 |
| **Greenhouse-Geisser** | 0.8660 | 0.0009 |
| **Box's Conservative** | 0.3333 | 0.0156 |

**Interpretation:**
- ✅ Sphericity assumption is reasonably met (ε ≥ 0.75)
- ✅ Results remain significant even with most conservative correction
- **All adjusted p-values < 0.05**, confirming robust time effect

---

## 3. Marginal Means Analysis

### Calculate Means with 98% Confidence Intervals

```stata
margins time_n
```

**Output:**

```
Predictive margins                                         Number of obs = 100

Expression: Linear prediction, predict()

------------------------------------------------------------------------------
             |            Delta-method
             |     Margin   std. err.      t    P>|t|     [98% conf. interval]
-------------+----------------------------------------------------------------
      time_n |
   Baseline  |    72.3192    .990068    73.04   0.000     69.96357    74.67483
    Month 1  |     70.264    .990068    70.97   0.000     67.90837    72.61963
    Month 2  |    66.8772    .990068    67.55   0.000     64.52157    69.23283
    Month 3  |    67.2868    .990068    67.96   0.000     64.93117    69.64243
------------------------------------------------------------------------------
```

### Summary Table of Means

| Time Point | Mean | SE | 98% CI Lower | 98% CI Upper | Change from Baseline |
|------------|------|----|--------------|--------------|--------------------|
| **Baseline** | 72.32 | 0.99 | 69.96 | 74.67 | — |
| **Month 1** | 70.26 | 0.99 | 67.91 | 72.62 | -2.06 |
| **Month 2** | 66.88 | 0.99 | 64.52 | 69.23 | **-5.44** |
| **Month 3** | 67.29 | 0.99 | 64.93 | 69.64 | -5.03 |

**Key Observations:**
- 📉 **Progressive decline from Baseline to Month 2** (5.44-point decrease)
- 📈 **Slight recovery at Month 3** (0.41-point increase from Month 2)
- 🎯 **Largest effect at Month 2**
- ⚠️ All 98% CIs are non-overlapping between Baseline and Months 2-3

---

## 4. Pairwise Comparisons (Bonferroni Corrected)

### Post-hoc Tests

```stata
pwcompare time_n, mcompare(bonferroni)
```

**Output:**

```
Pairwise comparisons of marginal linear predictions

Margins: asbalanced

---------------------------
             |    Number of
             |  comparisons
-------------+-------------
      time_n |            6
---------------------------

----------------------------------------------------------------------
                     |                                 Bonferroni
                     |   Contrast   Std. err.     [98% conf. interval]
---------------------+------------------------------------------------
              time_n |
Month 1 vs Baseline  |    -2.0552   1.400168     -6.306512    2.196112
Month 2 vs Baseline  |     -5.442   1.400168     -9.693312   -1.190688
Month 3 vs Baseline  |    -5.0324   1.400168     -9.283712   -.7810879
 Month 2 vs Month 1  |  -3.386801   1.400168     -7.638113    .8645115
 Month 3 vs Month 1  |    -2.9772   1.400168     -7.228512    1.274112
 Month 3 vs Month 2  |   .4096002   1.400168     -3.841712    4.660912
----------------------------------------------------------------------
```

### Detailed Pairwise Comparison Results

| Comparison | Difference | SE | 98% CI Lower | 98% CI Upper | Significant? | Effect |
|------------|-----------|----|--------------|--------------|--------------| -------|
| Month 1 vs Baseline | -2.06 | 1.40 | -6.31 | 2.20 | ❌ No | Small decline |
| **Month 2 vs Baseline** | **-5.44** | 1.40 | **-9.69** | **-1.19** | ✅ **Yes** | **Significant decrease** |
| **Month 3 vs Baseline** | **-5.03** | 1.40 | **-9.28** | **-0.78** | ✅ **Yes** | **Significant decrease** |
| Month 2 vs Month 1 | -3.39 | 1.40 | -7.64 | 0.86 | ❌ No | Decline continues |
| Month 3 vs Month 1 | -2.98 | 1.40 | -7.23 | 1.27 | ❌ No | Decline continues |
| Month 3 vs Month 2 | 0.41 | 1.40 | -3.84 | 4.66 | ❌ No | Slight recovery |

**Key Findings:**
- ✅ **Month 2 vs Baseline**: Significantly different (p < 0.02, Bonferroni corrected)
- ✅ **Month 3 vs Baseline**: Significantly different (p < 0.02, Bonferroni corrected)
- ❌ **Month 1 vs Baseline**: Not significant at 98% confidence level
- ❌ **Adjacent time points**: No significant differences between consecutive months
- 🎯 **Pattern**: The effect emerges by Month 2 and persists through Month 3

---

## 5. Visualizations

### Create Marginal Means Plot

```stata
margins time_n
marginsplot, recast(line) recastci(rarea) title("Values Over Time (98% CI)") ytitle("Value") xtitle("Time Point")
graph export "D:\Cognitive sience\Summer404\Statistic in Research\Prism\DPLFC\ANOVA\Simple line plot.png", as(png) name("Graph")
```

**Output:**
```
Variables that uniquely identify margins: time_n
file D:\...\Simple line plot.png saved as PNG format
```

**Graph Description:**
- Line plot showing mean values across four time points
- Shaded area represents 98% confidence intervals
- Clear visualization of decline from Baseline to Month 2
- Slight uptick from Month 2 to Month 3

### Create Individual Trajectories Plot

```stata
twoway line value time_n, connect(ascending) by(subject_id)
```

**Graph Description:**
- Spaghetti plot showing all 25 individual subject trajectories
- Reveals heterogeneity in individual responses
- Some subjects show decline, others remain stable or increase
- Panel layout with separate plot for each subject

### Create Box Plot Distribution

```stata
graph box value, over(time_n) title("Distribution of Values by Time Point")
graph export "D:\Cognitive sience\Summer404\Statistic in Research\Prism\DPLFC\ANOVA\Box plots by time.png", as(png) name("Graph")
```

**Output:**
```
file D:\...\Box plots by time.png saved as PNG format
```

**Graph Description:**
- Box plots showing distribution at each time point
- Displays median, quartiles, and outliers
- Shows decreasing median trend from Baseline to Month 2
- Reveals variability within each time point

---

## 6. Statistical Summary

### Study Design
- **Design**: Repeated Measures ANOVA (within-subjects)
- **Sample Size**: 25 subjects
- **Time Points**: 4 (Baseline, Month 1, Month 2, Month 3)
- **Total Observations**: 100
- **Confidence Level**: 98% (α = 0.02)

### Model Performance
- **R²**: 0.9082 (excellent fit)
- **Adjusted R²**: 0.8738
- **RMSE**: 4.95

### Main Results
1. **Omnibus Time Effect**: F(3, 72) = 6.77, **p = 0.0004***
2. **Effect Size**: Partial η² ≈ 0.22 (medium to large effect)
3. **Sphericity**: Reasonably met (ε = 0.87-0.98)

### Post-hoc Comparisons (Bonferroni Corrected, 98% CI)
- **Month 2 vs Baseline**: -5.44, 98% CI [-9.69, -1.19], **p < 0.02**
- **Month 3 vs Baseline**: -5.03, 98% CI [-9.28, -0.78], **p < 0.02**
- All other comparisons: Not significant at α = 0.02

---

## 7. Clinical/Scientific Interpretation

### Pattern of Change
1. **Baseline (Week 0)**: Mean = 72.32
2. **Month 1**: Small non-significant decline (Mean = 70.26, -2.06 points)
3. **Month 2**: Significant decline (Mean = 66.88, **-5.44 points from baseline**)
4. **Month 3**: Maintains low level (Mean = 67.29, **-5.03 points from baseline**)

### Key Findings
- ✅ **Statistically significant time effect** detected
- 📉 **Values decrease significantly by Month 2** and remain low through Month 3
- 🔄 **Slight recovery** from Month 2 to Month 3, but not back to baseline
- 📊 **Effect emerges gradually**: Not significant at Month 1, significant by Month 2

### Effect Magnitude
- **Absolute change**: 5.44 points (7.5% decrease from baseline)
- **Cohen's d** (estimated): ~0.50 (medium effect size)
- **Clinical significance**: Depends on measurement scale and context

### Variability
- High between-subject variability (F = 28.83, p < 0.0001)
- Individual trajectories show heterogeneous responses
- Some subjects show larger declines, others minimal change

---

## 8. Assumptions Check

### Sphericity
- ✅ **Huynh-Feldt ε = 0.98** (close to 1.0, sphericity met)
- ✅ **Greenhouse-Geisser ε = 0.87** (≥ 0.75, acceptable)
- ✅ Results remain significant with all corrections

### Normality
- Should verify with: `histogram residuals, normal` and `qnorm residuals`
- Visual inspection of residual plots recommended

### Independence
- ✅ Between-subject independence assumed (different subjects)
- ✅ Within-subject correlation properly modeled by repeated measures design

---

## 9. Conclusions

### Primary Conclusion
**There is strong statistical evidence (p = 0.0004) of a significant time effect on values across the four measurement periods. Values decline significantly from baseline to Month 2 (-5.44 points, 98% CI: -9.69 to -1.19, p < 0.02) and remain significantly reduced at Month 3 (-5.03 points, 98% CI: -9.28 to -0.78, p < 0.02).**

### Clinical Implications
1. The effect is not immediate (Month 1 shows no significant change)
2. Significant reduction emerges by Month 2
3. The effect persists through Month 3 with slight recovery
4. Individual variability is substantial

### Recommendations for Future Analysis
1. **Trend analysis**: Test for linear/quadratic trends
2. **Effect size calculation**: Compute Cohen's d for each comparison
3. **Responder analysis**: Identify subject characteristics predicting larger vs. smaller changes
4. **Missing data check**: Verify no systematic dropout patterns
5. **Sensitivity analysis**: Test robustness with different covariance structures

---

## 10. Complete Stata Code

```stata
*===============================================
* COMPLETE REPEATED MEASURES ANOVA ANALYSIS
* 98% Confidence Intervals
*===============================================

* Import data
import delimited "D:\Cognitive sience\Summer404\Statistic in Research\Prism\DPLFC\ANOVA\repeated_anova_p005_long.csv", clear

* Create numeric time variable
encode time, gen(time_num)
gen time_n = .
replace time_n = 0 if time == "Baseline"
replace time_n = 1 if time == "Month_1"
replace time_n = 2 if time == "Month_2"
replace time_n = 3 if time == "Month_3"

* Label time variable
label define timelbl 0 "Baseline" 1 "Month 1" 2 "Month 2" 3 "Month 3"
label values time_n timelbl

* Verify data structure
summarize
list in 1/10

* Run repeated measures ANOVA
anova value subject_id time_n, repeated(time_n)

* Marginal means with 98% CI
margins time_n

* Pairwise comparisons (Bonferroni corrected)
pwcompare time_n, mcompare(bonferroni)

* Visualizations
margins time_n
marginsplot, recast(line) recastci(rarea) title("Values Over Time (98% CI)") ytitle("Value") xtitle("Time Point")
graph export "Simple_line_plot.png", as(png) replace

twoway line value time_n, connect(ascending) by(subject_id)
graph export "Individual_trajectories.png", as(png) replace

graph box value, over(time_n) title("Distribution of Values by Time Point")
graph export "Box_plots_by_time.png", as(png) replace

* Save log
log close
```