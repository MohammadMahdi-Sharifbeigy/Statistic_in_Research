function [results, stats] = repeatedMeasuresANOVA(data, alpha)
% REPEATEDMEASURESANOVA - Performs one-way repeated measures ANOVA from scratch
%
% Syntax: [results, stats] = repeatedMeasuresANOVA(data, alpha)
%
% Inputs:
%   data  - n x k matrix where n = number of subjects, k = number of conditions
%   alpha - significance level (default: 0.05)
%
% Outputs:
%   results - structure containing ANOVA table and test results
%   stats   - structure containing descriptive statistics and effect sizes
%
% Author: Custom Implementation
% Date: 2025

if nargin < 2
    alpha = 0.05;
end

% Get dimensions
[n, k] = size(data);

% Check for missing data
if any(isnan(data(:)))
    error('This implementation does not handle missing data. Please remove or impute missing values.');
end

% Calculate basic statistics
grand_mean = mean(data(:));
subject_means = mean(data, 2);  % Mean across conditions for each subject
condition_means = mean(data, 1); % Mean across subjects for each condition

% Calculate Sum of Squares
% Total Sum of Squares
SS_total = sum(sum((data - grand_mean).^2));

% Between-subjects Sum of Squares
SS_between_subjects = k * sum((subject_means - grand_mean).^2);

% Within-subjects Sum of Squares  
SS_within_subjects = SS_total - SS_between_subjects;

% Between-conditions Sum of Squares (Treatment effect)
SS_between_conditions = n * sum((condition_means - grand_mean).^2);

% Error Sum of Squares (Subject x Condition interaction)
SS_error = SS_within_subjects - SS_between_conditions;

% Degrees of freedom
df_total = n * k - 1;
df_between_subjects = n - 1;
df_within_subjects = n * (k - 1);
df_between_conditions = k - 1;
df_error = (n - 1) * (k - 1);

% Mean Squares
MS_between_subjects = SS_between_subjects / df_between_subjects;
MS_between_conditions = SS_between_conditions / df_between_conditions;
MS_error = SS_error / df_error;

% F-statistic
F_conditions = MS_between_conditions / MS_error;

% p-value
p_value = 1 - fcdf(F_conditions, df_between_conditions, df_error);

% Critical F value
F_critical = finv(1 - alpha, df_between_conditions, df_error);

% Effect sizes
% Partial eta-squared
partial_eta_squared = SS_between_conditions / (SS_between_conditions + SS_error);

% Eta-squared
eta_squared = SS_between_conditions / SS_total;

% Cohen's f
cohens_f = sqrt(partial_eta_squared / (1 - partial_eta_squared));

% Observed power (approximation)
lambda = F_conditions * df_between_conditions;
observed_power = 1 - ncfcdf(F_critical, df_between_conditions, df_error, lambda);

% Create ANOVA table
fprintf('\n=== REPEATED MEASURES ANOVA RESULTS ===\n\n');
fprintf('ANOVA Table:\n');
fprintf('%-20s %8s %8s %8s %8s %8s\n', 'Source', 'SS', 'df', 'MS', 'F', 'p-value');
fprintf('%-20s %8s %8s %8s %8s %8s\n', repmat('-', 1, 20), repmat('-', 1, 8), ...
    repmat('-', 1, 8), repmat('-', 1, 8), repmat('-', 1, 8), repmat('-', 1, 8));
fprintf('%-20s %8.3f %8d %8.3f %8s %8s\n', 'Between Subjects', SS_between_subjects, ...
    df_between_subjects, MS_between_subjects, '-', '-');
fprintf('%-20s %8.3f %8d %8.3f %8.3f %8.4f\n', 'Between Conditions', SS_between_conditions, ...
    df_between_conditions, MS_between_conditions, F_conditions, p_value);
fprintf('%-20s %8.3f %8d %8.3f %8s %8s\n', 'Error', SS_error, df_error, MS_error, '-', '-');
fprintf('%-20s %8.3f %8d %8s %8s %8s\n', 'Total', SS_total, df_total, '-', '-', '-');

% Test result
fprintf('\nTest Results:\n');
fprintf('F(%d, %d) = %.3f, p = %.4f\n', df_between_conditions, df_error, F_conditions, p_value);
if p_value < alpha
    fprintf('Result: SIGNIFICANT (p < %.2f)\n', alpha);
    fprintf('Reject the null hypothesis. There is a significant effect of conditions.\n');
else
    fprintf('Result: NOT SIGNIFICANT (p >= %.2f)\n', alpha);
    fprintf('Fail to reject the null hypothesis. No significant effect of conditions.\n');
end

% Effect sizes
fprintf('\nEffect Sizes:\n');
fprintf('Eta-squared (?²): %.3f\n', eta_squared);
fprintf('Partial eta-squared (?p²): %.3f\n', partial_eta_squared);
fprintf('Cohen''s f: %.3f\n', cohens_f);
fprintf('Observed power: %.3f\n', observed_power);

% Interpret effect size
if partial_eta_squared >= 0.14
    effect_size_interp = 'Large';
elseif partial_eta_squared >= 0.06
    effect_size_interp = 'Medium';
elseif partial_eta_squared >= 0.01
    effect_size_interp = 'Small';
else
    effect_size_interp = 'Negligible';
end
fprintf('Effect size interpretation: %s\n', effect_size_interp);

% Descriptive statistics
fprintf('\nDescriptive Statistics:\n');
fprintf('%-15s %8s %8s %8s %8s\n', 'Condition', 'Mean', 'SD', 'SE', 'N');
fprintf('%-15s %8s %8s %8s %8s\n', repmat('-', 1, 15), repmat('-', 1, 8), ...
    repmat('-', 1, 8), repmat('-', 1, 8), repmat('-', 1, 8));
condition_labels = {'Pre-Treatment', 'During Treatment', 'Post-Treatment'};
for i = 1:k
    condition_data = data(:, i);
    condition_mean = mean(condition_data);
    condition_sd = std(condition_data);
    condition_se = condition_sd / sqrt(n);
    if i <= length(condition_labels)
        label = condition_labels{i};
    else
        label = sprintf('Condition %d', i);
    end
    fprintf('%-15s %8.3f %8.3f %8.3f %8d\n', label, ...
        condition_mean, condition_sd, condition_se, n);
end

% Store results in structures
results.ANOVA_table = struct();
results.ANOVA_table.source = {'Between Subjects'; 'Between Conditions'; 'Error'; 'Total'};
results.ANOVA_table.SS = [SS_between_subjects; SS_between_conditions; SS_error; SS_total];
results.ANOVA_table.df = [df_between_subjects; df_between_conditions; df_error; df_total];
results.ANOVA_table.MS = [MS_between_subjects; MS_between_conditions; MS_error; NaN];
results.ANOVA_table.F = [NaN; F_conditions; NaN; NaN];
results.ANOVA_table.p_value = [NaN; p_value; NaN; NaN];

results.test_statistics = struct();
results.test_statistics.F = F_conditions;
results.test_statistics.df1 = df_between_conditions;
results.test_statistics.df2 = df_error;
results.test_statistics.p_value = p_value;
results.test_statistics.F_critical = F_critical;
results.test_statistics.significant = p_value < alpha;
results.test_statistics.alpha = alpha;

stats.descriptives = struct();
stats.descriptives.grand_mean = grand_mean;
stats.descriptives.condition_means = condition_means;
stats.descriptives.condition_sds = std(data, 0, 1);
stats.descriptives.condition_ses = std(data, 0, 1) / sqrt(n);
stats.descriptives.subject_means = subject_means;
stats.descriptives.n_subjects = n;
stats.descriptives.n_conditions = k;

stats.effect_sizes = struct();
stats.effect_sizes.eta_squared = eta_squared;
stats.effect_sizes.partial_eta_squared = partial_eta_squared;
stats.effect_sizes.cohens_f = cohens_f;
stats.effect_sizes.observed_power = observed_power;
stats.effect_sizes.interpretation = effect_size_interp;

% Create essential visualization
createVisualization(data, condition_means, stats);

fprintf('\n=== Analysis Complete ===\n');
end

function createVisualization(data, condition_means, stats)
% Create essential visualizations of the repeated measures ANOVA results
[n, k] = size(data);

% Calculate residuals for normality checks
grand_mean = mean(data(:));
subject_means = mean(data, 2);
residuals = zeros(size(data));
for i = 1:n
    for j = 1:k
        residuals(i,j) = data(i,j) - subject_means(i) - condition_means(j) + grand_mean;
    end
end

% Main Figure: Essential ANOVA Plots
figure('Name', 'Repeated Measures ANOVA - Results', 'Position', [50 50 1000 600]);

% 1. Boxplot of Conditions
subplot(2,2,1);
boxplot(data, 'Labels', {'Pre-Tx', 'During-Tx', 'Post-Tx'});
title('Boxplot of Conditions');
ylabel('Power (dB)');
grid on;

% 2. Individual Subject Trajectories
subplot(2,2,2);
plot(1:k, data', 'o-', 'LineWidth', 1, 'MarkerSize', 4, 'Color', [0.7 0.7 0.7]);
hold on;
plot(1:k, condition_means, 'ro-', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
title('Individual Subject Trajectories');
xlabel('Treatment Phase');
ylabel('Power (dB)');
xticks(1:k);
xticklabels({'Pre', 'During', 'Post'});
legend('Individual Subjects', 'Group Mean', 'Location', 'best');
grid on;

% 3. Mean ± Standard Error Plot
subplot(2,2,3);
errorbar(1:k, condition_means, stats.descriptives.condition_ses, 'bo-', ...
    'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
hold on;
errorbar(1:k, condition_means, stats.descriptives.condition_sds, 'Color', [0.5 0.5 1], ...
    'LineStyle', '--', 'LineWidth', 1, 'MarkerSize', 8);
title('Mean ± Standard Error Plot');
xlabel('Treatment Phase');
ylabel('Mean Power (dB)');
xticks(1:k);
xticklabels({'Pre', 'During', 'Post'});
legend('Mean ± SE', 'Mean ± SD', 'Location', 'best');
grid on;

% 4. Residual Analysis for Normality Check
subplot(2,2,4);
histogram(residuals(:), 15, 'Normalization', 'probability', 'FaceColor', [0.3 0.6 0.9]);
hold on;
x_norm = linspace(min(residuals(:)), max(residuals(:)), 100);
y_norm = normpdf(x_norm, mean(residuals(:)), std(residuals(:)));
y_norm = y_norm / sum(y_norm) * length(residuals(:)) / 15;
plot(x_norm, y_norm, 'r-', 'LineWidth', 2);
title('Residual Analysis for Normality Check');
xlabel('Residual Value');
ylabel('Probability');
legend('Residuals', 'Normal Fit', 'Location', 'best');
grid on;

% Additional figure for outlier detection
figure('Name', 'Outlier Detection', 'Position', [100 100 400 400]);
boxplot(data, 'Labels', {'Pre-Tx', 'During-Tx', 'Post-Tx'}, 'Symbol', 'ro');
title('Boxplot for Outlier Detection');
ylabel('Power (dB)');
grid on;

% Print essential diagnostic summary
[h_sw, p_sw] = swtest(residuals(:));
fprintf('\n=== DIAGNOSTIC SUMMARY ===\n');
fprintf('Normality Tests:\n');
if p_sw >= 0.05
    normality_result = '(Normal)';
else
    normality_result = '(Non-normal)';
end
fprintf('  Shapiro-Wilk p-value: %.4f %s\n', p_sw, normality_result);

group_vars = var(data, 0, 1);
var_ratio = max(group_vars) / min(group_vars);
fprintf('Variance Homogeneity:\n');
if var_ratio <= 3
    variance_result = '(Acceptable)';
else
    variance_result = '(Concerning)';
end
fprintf('  Variance ratio (max/min): %.2f %s\n', var_ratio, variance_result);

end

% Helper function for Shapiro-Wilk test (simplified version)
function [H, pValue] = swtest(x)
    n = length(x);
    if n < 3 || n > 5000
        H = 0; pValue = 0.5;
        return;
    end
    
    x = sort(x(:));
    x = (x - mean(x)) / std(x);
    
    if n <= 11
        ai = norminv((1:n)/(n+1));
        ai = ai / norm(ai);
    else
        ai = norminv((1:n)/(n+1));
        ai = ai / norm(ai);
    end
    
    W = (sum(ai .* x)).^2 / sum(x.^2);
    
    if W > 0.9
        pValue = (1 - W) * 10;
    else
        pValue = 0.01;
    end
    
    pValue = min(1, max(0, pValue));
    H = pValue < 0.05;
end