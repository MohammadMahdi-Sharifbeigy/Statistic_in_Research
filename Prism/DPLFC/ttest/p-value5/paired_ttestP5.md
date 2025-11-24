# Paired T-Test Analysis Report
## Before vs. After Treatment Comparison (98% Confidence Intervals)

---

## 1. Analysis Overview

### Study Design
- **Test Type**: Paired (dependent) t-test
- **Sample Size**: 60 paired observations
- **Confidence Level**: 98% (α = 0.02)
- **Comparison**: Before Treatment vs. After Treatment (same subjects measured twice)

### Research Question
**Does the treatment significantly change the measured values?**

---

## 2. Statistical Analysis

### Run Paired T-Test

```stata
ttest before_treatment == after_treatment
```

### Full Output

```
Paired t test
------------------------------------------------------------------------------
Variable |     Obs        Mean    Std. err.   Std. dev.   [98% conf. interval]
---------+--------------------------------------------------------------------
before~t |      60     84.9135    1.528687    11.84116    81.25806    88.56894
after_~t |      60    79.79933    1.631363    12.63648    75.89837     83.7003
---------+--------------------------------------------------------------------
    diff |      60    5.114166    .7562506    5.857892    3.305798    6.922534
------------------------------------------------------------------------------
     mean(diff) = mean(before_treatment - after_treatment)        t =   6.7625
 H0: mean(diff) = 0                              Degrees of freedom =       59
 Ha: mean(diff) < 0           Ha: mean(diff) != 0           Ha: mean(diff) > 0
 Pr(T < t) = 1.0000         Pr(|T| > |t|) = 0.0000          Pr(T > t) = 0.0000
```

---

## 3. Descriptive Statistics

### Summary Table

| Variable | N | Mean | Std. Error | Std. Dev. | 98% CI Lower | 98% CI Upper |
|----------|---|------|------------|-----------|--------------|--------------|
| **Before Treatment** | 60 | 84.91 | 1.53 | 11.84 | 81.26 | 88.57 |
| **After Treatment** | 60 | 79.80 | 1.63 | 12.64 | 75.90 | 83.70 |
| **Difference** | 60 | **5.11** | 0.76 | 5.86 | **3.31** | **6.92** |

### Key Observations
- 📊 **Before Treatment**: Mean = 84.91 (SD = 11.84)
- 📊 **After Treatment**: Mean = 79.80 (SD = 12.64)
- 📉 **Mean Decrease**: 5.11 points (6.0% reduction)
- ✅ **98% CI of difference**: [3.31, 6.92] (does NOT include zero)

---

## 4. Hypothesis Test Results

### Hypotheses

**Null Hypothesis (H₀)**: mean(difference) = 0  
*There is no difference between before and after treatment*

**Alternative Hypothesis (Hₐ)**: mean(difference) ≠ 0  
*There is a significant difference between before and after treatment*

### Test Statistics

| Statistic | Value |
|-----------|-------|
| **Mean Difference** | 5.11 points |
| **Standard Error of Difference** | 0.76 |
| **t-statistic** | **6.7625** |
| **Degrees of Freedom** | 59 |
| **p-value (two-tailed)** | **< 0.0001*** |

### Decision

✅ **REJECT the null hypothesis**

**Conclusion**: There is **highly significant evidence** (t = 6.76, df = 59, p < 0.0001) that treatment significantly reduces values. At the 98% confidence level, the treatment causes a mean reduction of 5.11 points (98% CI: 3.31 to 6.92).

---

## 5. Interpretation

### Statistical Significance
- **p-value < 0.0001**: Extremely strong evidence against null hypothesis
- **Significant at α = 0.02**: The result exceeds the 98% confidence threshold
- **Significant at α = 0.01**: The result also exceeds the 99% confidence threshold
- **Significant at α = 0.001**: The result is significant even at 99.9% confidence

### Effect Size

#### Raw Effect
- **Absolute change**: 5.11 points decrease
- **Percentage change**: 6.0% reduction from baseline
- **98% CI**: [3.31, 6.92] - very precise estimate

#### Cohen's d (Effect Size)
```
Cohen's d = Mean difference / SD of difference
Cohen's d = 5.11 / 5.86 = 0.87
```

**Interpretation**: **Large effect size** (d > 0.8 is considered large)

| Effect Size | Cohen's d | Interpretation |
|-------------|-----------|----------------|
| Small | 0.2 | |
| Medium | 0.5 | |
| **Large** | **0.8** | **✓ Our result: 0.87** |

### Clinical/Practical Significance
- The treatment produces a **statistically significant** reduction
- The effect size is **large and clinically meaningful**
- The effect is **consistent** (narrow confidence interval)
- The reduction of ~5 points represents a **substantial change** depending on the measurement scale

---

## 6. Directional Tests

The output shows three alternative hypotheses:

### Two-Tailed Test (Default)
**Hₐ: mean(diff) ≠ 0**  
**p-value = 0.0000** ✅ **Significant**

*Tests if there's ANY difference (increase or decrease)*

### Right-Tailed Test
**Hₐ: mean(diff) > 0** (Before > After, i.e., treatment decreases values)  
**p-value = 0.0000** ✅ **Significant**

*Tests if treatment DECREASES values - This is what we observe!*

### Left-Tailed Test
**Hₐ: mean(diff) < 0** (Before < After, i.e., treatment increases values)  
**p-value = 1.0000** ❌ Not significant

*Tests if treatment INCREASES values - Not supported by data*

**Conclusion**: The treatment **significantly decreases** values (p < 0.0001).

---

## 7. Visualization

### Create Box Plot

```stata
graph box before_treatment after_treatment, title("Paired T-Test") ytitle("Value") legend(off)
graph export "D:\Cognitive sience\Summer404\Statistic in Research\Prism\DPLFC\ttest\p-value5\t-testp5.png", as(png) name("Graph")
```

**Output:**
```
file D:\...\t-testp5.png saved as PNG format
```

### Graph Description
- **Box plot** showing distribution of values before and after treatment
- **Before Treatment Box**:
  - Median and quartiles visible
  - Shows spread of baseline values
- **After Treatment Box**:
  - Lower position indicates decreased values
  - Similar spread suggests consistent effect
- **Visual Comparison**:
  - Clear downward shift from before to after
  - Overlap between boxes shows some variability
  - Overall pattern confirms treatment effect

---

## 8. Assumptions Check

### Paired T-Test Assumptions

#### 1. ✅ Paired Data
- Same 60 subjects measured twice
- Before and after measurements on same individuals
- **Assumption met**

#### 2. ✅ Random Sample
- Assumes subjects randomly selected from population
- Should be verified by study design
- **Check with study protocol**

#### 3. 🔍 Normality of Differences
- Required: Differences should be approximately normally distributed
- **Recommen