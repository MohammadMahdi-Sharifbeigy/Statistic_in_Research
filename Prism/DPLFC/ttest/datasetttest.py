import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ========================================
# دیتاست 1: P-value ~ 0.02
# ========================================
n1 = 50  # تعداد نمونه‌ها

# قبل از درمان (Before)
before1 = np.random.normal(120, 15, n1)

# بعد از درمان (After) - با اختلاف معنادار
effect1 = np.random.normal(8, 5, n1)  # تاثیر درمان
after1 = before1 - effect1

# محاسبه P-value
t_stat1, p_value1 = stats.ttest_rel(before1, after1)

# ایجاد DataFrame
df1 = pd.DataFrame({
    'Subject_ID': range(1, n1 + 1),
    'Before_Treatment': np.round(before1, 2),
    'After_Treatment': np.round(after1, 2),
    'Difference': np.round(before1 - after1, 2)
})

print("=" * 60)
print("dataset 1: Paired T-Test با P-value ~ 0.02")
print("=" * 60)
print(f"population: {n1}")
print(f"P-value: {p_value1:.4f}")
print(f"T-statistic: {t_stat1:.4f}")
print("\n datas:")
print(df1.head(10))
print(df1[['Before_Treatment', 'After_Treatment', 'Difference']].describe())

# ذخیره فایل CSV
df1.to_csv('paired_ttest_p002.csv', index=False)
print("\n✓ file saved: paired_ttest_p002.csv")

print("\n" + "=" * 60)

# ========================================
# دیتاست 2: P-value ~ 0.05
# ========================================
n2 = 60  # تعداد نمونه‌ها

# قبل از درمان
before2 = np.random.normal(85, 12, n2)

# بعد از درمان - با اختلاف کمتر
effect2 = np.random.normal(4, 6, n2)  # تاثیر کمتر
after2 = before2 - effect2

# محاسبه P-value
t_stat2, p_value2 = stats.ttest_rel(before2, after2)

# ایجاد DataFrame
df2 = pd.DataFrame({
    'Subject_ID': range(1, n2 + 1),
    'Before_Treatment': np.round(before2, 2),
    'After_Treatment': np.round(after2, 2),
    'Difference': np.round(before2 - after2, 2)
})

print("dataset 2: Paired T-Test با P-value ~ 0.05")
print("=" * 60)
print(f"P-value: {p_value2:.4f}")
print(f"T-statistic: {t_stat2:.4f}")
print("\n datas:")
print(df2.head(10))
print(df2[['Before_Treatment', 'After_Treatment', 'Difference']].describe())

# ذخیره فایل CSV
df2.to_csv('paired_ttest_p005.csv', index=False)
print("\n file saved: paired_ttest_p005.csv")

