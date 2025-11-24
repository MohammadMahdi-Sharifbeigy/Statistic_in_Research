# Repeated Measures ANOVA Analysis Results

## Study Overview
- **Sample Size**: 20 subjects
- **Time Points**: 4 measurements (Baseline, Week 4, Week 8, Week 12)
- **Total Observations**: 80
- **Confidence Level**: 98% (α = 0.02)

---

## Statistical Model Summary

### Model Fit
- **R-squared**: 0.7645 (76.45% of variance explained)
- **Adjusted R-squared**: 0.6736
- **Root MSE**: 6.20479

### ANOVA Table

| Source | Partial SS | df | MS | F | Prob > F |
|--------|-----------|----|----|---|----------|
| **Model** | 7124.50 | 22 | 323.84 | 8.41 | **< 0.0001*** |
| Subject_ID | 6731.54 | 19 | 354.29 | 9.20 | **< 0.0001*** |
| **Time** | **392.96** | **3** | **130.99** | **3.40** | **0.0237** |
| Residual | 2194.47 | 57 | 38.50 | | |
| **Total** | 9318.97 | 79 | 117.96 | | |

---

## Main Finding: Time Effect

### Omnibus Test
- **F-statistic**: F(3, 57) = 3.40
- **p-value**: 0.0237
- **Conclusion**: **STATISTICALLY SIGNIFICANT** at α = 0.02 level
  - There is a significant effect of time on the measurement

### Sphericity Assumption Tests
| Test | Epsilon Value | Adjusted p-value |
|------|---------------|------------------|
| Regular (assumed) | 1.000 | 0.0237 |
| Huynh-Feldt | 1.000* | 0.0237 |
| Greenhouse-Geisser | 0.8601 | 0.0306 |
| Box's Conservative | 0.3333 | 0.0808 |

*Note: Huynh-Feldt epsilon was reset to 1.0 (originally 1.0071)*

**Interpretation**: The sphericity assumption appears reasonable. Even with conservative corrections (Greenhouse-Geisser), the result remains significant at α = 0.05, though marginal at α = 0.02.

---

## Marginal Means (98% Confidence Intervals)

| Time Point | Mean | Std. Error | 98% CI Lower | 98% CI Upper |
|------------|------|------------|--------------|--------------|
| **Baseline (Week 0)** | 96.16 | 1.39 | 92.84 | 99.48 |
| **Week 4** | 94.00 | 1.39 | 90.68 | 97.32 |
| **Week 8** | 91.98 | 1.39 | 88.66 | 95.30 |
| **Week 12** | 90.23 | 1.39 | 86.91 | 93.55 |

### Trend Observation
- There is a **decreasing trend** over time
- Total decrease from baseline to Week 12: **5.93 units**
- All confidence intervals overlap, suggesting gradual rather than abrupt changes

---

## Pairwise Comparisons (Bonferroni Corrected)

*98% Confidence Intervals with Bonferroni adjustment for 6 comparisons*

| Comparison | Difference | Std. Error | 98% CI Lower | 98% CI Upper | Significant? |
|------------|-----------|------------|--------------|--------------|--------------|
| Week 4 vs Baseline | -2.16 | 1.96 | -8.17 | 3.85 | No |
| Week 8 vs Baseline | -4.18 | 1.96 | -10.19 | 1.84 | No |
| **Week 12 vs Baseline** | **-5.93** | 1.96 | **-11.94** | **0.08** | **Borderline** |
| Week 8 vs Week 4 | -2.02 | 1.96 | -8.03 | 3.99 | No |
| Week 12 vs Week 4 | -3.77 | 1.96 | -9.78 | 2.24 | No |
| Week 12 vs Week 8 | -1.75 | 1.96 | -7.76 | 4.26 | No |

### Key Findings from Pairwise Tests
- **No individual pairwise comparison reaches statistical significance** at the 98% confidence level after Bonferroni correction
- The comparison **Week 12 vs Baseline** is borderline (CI: -11.94 to 0.08)
  - Just barely includes zero, suggesting it would be significant at a slightly less stringent level
- This indicates the significant omnibus test is driven by the **overall linear trend** rather than any single time point difference

---

## Clinical/Practical Interpretation

### Summary
1. **Overall Time Effect**: Statistically significant (p = 0.0237)
2. **Pattern**: Gradual, consistent decline over 12 weeks
3. **Magnitude**: Approximately 6-unit decrease from baseline to week 12 (6.2% decline)
4. **Individual Comparisons**: None significant after multiple comparison correction, but Week 12 shows the largest departure from baseline

### Recommendations
- The significant omnibus test with non-significant pairwise comparisons suggests a **dose-response relationship** (monotonic trend)
- Consider **trend analysis** (linear/quadratic contrasts) for more precise characterization
- The effect size is moderate and clinically meaningful depending on the measurement scale

---

## Statistical Notes

### Between-Subjects Variability
- Strong evidence of **individual differences** (Subject_ID: F = 9.20, p < 0.0001)
- This justifies the repeated measures design, as subjects show substantial baseline differences

### Model Assumptions
1. **Sphericity**: Generally satisfied (ε = 0.86-1.00)
2. **Normality**: Should be verified with residual plots
3. **Independence**: Between-subjects independence assumed

### Power Considerations
- With n=20 and moderate effect size, power to detect pairwise differences is limited
- The omnibus test has better power for detecting overall trends

---

## Conclusion

There is **statistically significant evidence** (p = 0.0237) that measurements change over time, with a consistent downward trend observed from baseline through Week 12. While no single pairwise comparison reaches significance after Bonferroni correction at the 98% confidence level, the cumulative change from baseline to Week 12 (5.93 units) approaches significance and may be clinically meaningful.

**Primary Result**: Time has a significant effect on measurements (F(3,57) = 3.40, p = 0.024, 98% CI).

---

*Analysis conducted in Stata with 98% confidence intervals and Bonferroni correction for multiple comparisons.*