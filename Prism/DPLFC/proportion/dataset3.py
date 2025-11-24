import numpy as np
import pandas as pd
from scipy import stats

# تنظیم seed
np.random.seed(456)

# ========================================
# دیتاست Proportion Test
# مقایسه نرخ موفقیت در دو گروه
# ========================================

# پارامترهای دو گروه
n_group1 = 150  # تعداد در گروه کنترل
n_group2 = 150  # تعداد در گروه درمان

# نرخ موفقیت
success_rate_group1 = 0.40  # 40% موفقیت در گروه کنترل
success_rate_group2 = 0.60  # 60% موفقیت در گروه درمان

# ایجاد داده‌های binary (0 = شکست، 1 = موفقیت)
group1_outcomes = np.random.binomial(1, success_rate_group1, n_group1)
group2_outcomes = np.random.binomial(1, success_rate_group2, n_group2)

# ایجاد DataFrame
df_group1 = pd.DataFrame({
    'Patient_ID': range(1, n_group1 + 1),
    'Group': 'Control',
    'Treatment_Success': group1_outcomes
})

df_group2 = pd.DataFrame({
    'Patient_ID': range(n_group1 + 1, n_group1 + n_group2 + 1),
    'Group': 'Treatment',
    'Treatment_Success': group2_outcomes
})

# ترکیب داده‌ها
df_proportion = pd.concat([df_group1, df_group2], ignore_index=True)

# محاسبه آماره‌ها
success_g1 = group1_outcomes.sum()
success_g2 = group2_outcomes.sum()
prop_g1 = success_g1 / n_group1
prop_g2 = success_g2 / n_group2

# انجام Two-proportion Z-test
from statsmodels.stats.proportion import proportions_ztest

counts = np.array([success_g1, success_g2])
nobs = np.array([n_group1, n_group2])
z_stat, p_value = proportions_ztest(counts, nobs)

print("=" * 70)
print("دیتاست Proportion Test - مقایسه نرخ موفقیت بین دو گروه")
print("=" * 70)
print(f"\nگروه کنترل:")
print(f"  تعداد کل: {n_group1}")
print(f"  تعداد موفقیت: {success_g1}")
print(f"  نرخ موفقیت: {prop_g1:.2%}")

print(f"\nگروه درمان:")
print(f"  تعداد کل: {n_group2}")
print(f"  تعداد موفقیت: {success_g2}")
print(f"  نرخ موفقیت: {prop_g2:.2%}")

print(f"\nنتایج آماری:")
print(f"  Z-statistic: {z_stat:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  تفاوت نرخ‌ها: {(prop_g2 - prop_g1):.2%}")

print("\nنمونه از داده‌ها:")
print(df_proportion.head(15))

print("\nتوزیع فراوانی:")
print(df_proportion.groupby(['Group', 'Treatment_Success']).size().unstack(fill_value=0))

# ذخیره فایل
df_proportion.to_csv('proportion_test_data.csv', index=False)
print("\n✓ فایل ذخیره شد: proportion_test_data.csv")

# ایجاد فایل خلاصه برای نمودار
summary_data = pd.DataFrame({
    'Group': ['Control', 'Treatment'],
    'Total': [n_group1, n_group2],
    'Success': [success_g1, success_g2],
    'Failure': [n_group1 - success_g1, n_group2 - success_g2],
    'Success_Rate': [prop_g1, prop_g2]
})

summary_data.to_csv('proportion_test_summary.csv', index=False)
print("✓ فایل خلاصه ذخیره شد: proportion_test_summary.csv")

print("\n" + "=" * 70)
print("راهنمای استفاده در Stata:")
print("=" * 70)
print("""
📊 روش 1: استفاده از داده‌های خام (proportion_test_data.csv)

* وارد کردن داده
import delimited "proportion_test_data.csv", clear

* تبدیل متغیر Group به عددی
encode group, gen(group_num)

* انجام Two-proportion Test
prtest treatment_success, by(group)

* جدول توزیع فراوانی
tabulate group treatment_success, chi2 row col

* رسم نمودار میله‌ای (Bar Chart)
graph bar (mean) treatment_success, over(group) ///
    ytitle("Success Rate") ///
    title("Treatment Success Rate by Group") ///
    ylabel(0(0.1)1, angle(horizontal)) ///
    blabel(bar, format(%4.2f))

* رسم نمودار دایره‌ای برای هر گروه
graph pie if group=="Control", over(treatment_success) ///
    title("Control Group") ///
    plabel(_all percent, format(%3.1f)) ///
    legend(label(1 "Failure") label(2 "Success"))
    
graph pie if group=="Treatment", over(treatment_success) ///
    title("Treatment Group") ///
    plabel(_all percent, format(%3.1f)) ///
    legend(label(1 "Failure") label(2 "Success"))

-----------------------------------------------------------

📊 روش 2: استفاده از داده‌های خلاصه (proportion_test_summary.csv)

* وارد کردن داده خلاصه
import delimited "proportion_test_summary.csv", clear

* رسم نمودار Stacked Bar Chart
graph bar success failure, over(group) stack ///
    title("Treatment Outcomes by Group") ///
    ytitle("Number of Patients") ///
    legend(label(1 "Success") label(2 "Failure"))

* رسم نمودار نرخ موفقیت با خطوط اطمینان
generate ci_lower = success_rate - 1.96*sqrt(success_rate*(1-success_rate)/total)
generate ci_upper = success_rate + 1.96*sqrt(success_rate*(1-success_rate)/total)

twoway (bar success_rate group, barwidth(0.5)) ///
       (rcap ci_lower ci_upper group), ///
    ytitle("Success Rate") ///
    xlabel(1 "Control" 2 "Treatment") ///
    title("Success Rate with 95% CI") ///
    ylabel(0(0.1)1, angle(horizontal)) ///
    legend(off)

* رسم نمودار Dot Plot
graph dot success_rate, over(group) ///
    marker(1, msize(large)) ///
    ytitle("Success Rate") ///
    title("Treatment Success Rate Comparison") ///
    ylabel(0(0.1)1, angle(horizontal))

-----------------------------------------------------------

💡 نکات مهم:
1. برای CI دقیق‌تر از دستور 'prtest' استفاده کنید
2. برای مقایسه بیش از 2 گروه از Chi-square test استفاده کنید
3. می‌توانید نمودارها را با گزینه 'scheme()' سفارشی‌سازی کنید
   مثال: scheme(s2color) یا scheme(economist)
""")

print("\n" + "=" * 70)
print("✅ تمام فایل‌ها با موفقیت ایجاد شدند!")
print("=" * 70)