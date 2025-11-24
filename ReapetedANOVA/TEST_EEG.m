function test_EEG_RepeatedMeasuresANOVA()
% TEST_EEG_REPEATEDMEASURESANOVA - Test script for EEG-like data analysis
% 
% This script generates realistic EEG-like data for 20 subjects across 3 
% time points: Pre-treatment, During treatment, and Post-treatment (24 months)
% The data includes realistic EEG characteristics and eye artifact noise
%
% Author: Custom Implementation
% Date: 2025

clc; clear; close all;

fprintf('=== EEG Repeated Measures ANOVA Test ===\n');
fprintf('Generating realistic EEG data with eye artifacts...\n\n');

%% Parameters for EEG-like data generation
n_subjects = 20;
n_conditions = 3; % Pre-treatment, During treatment, Post-treatment
fs = 250; % Sampling frequency (Hz)
duration = 2; % seconds of data per trial
n_samples = fs * duration;

% EEG frequency bands
alpha_freq = [8 13]; % Alpha band (8-13 Hz)
beta_freq = [13 30];  % Beta band (13-30 Hz)
theta_freq = [4 8];   % Theta band (4-8 Hz)
delta_freq = [0.5 4]; % Delta band (0.5-4 Hz)

% Treatment effect parameters
baseline_alpha_power = 10; % dB
treatment_effect = [0, -2.5, -4.0]; % Progressive improvement in alpha power
individual_variation = 2.0; % Individual differences in baseline

% Eye artifact parameters
eye_blink_freq = 0.5; % Average blinks per second
eye_artifact_amplitude = [15, 12, 8]; % Decreasing with treatment (better compliance)

%% Generate realistic EEG data for each subject and condition
fprintf('Generating data for %d subjects across %d conditions...\n', n_subjects, n_conditions);

% Initialize data matrix
eeg_alpha_power = zeros(n_subjects, n_conditions);
raw_eeg_data = cell(n_subjects, n_conditions);

% Set random seed for reproducibility
rng(42);

for subj = 1:n_subjects
    % Individual baseline characteristics
    subject_baseline = baseline_alpha_power + randn * individual_variation;
    subject_eye_tendency = 0.8 + 0.4 * rand; % Individual eye artifact tendency
    
    fprintf('Processing Subject %d/%d\n', subj, n_subjects);
    
    for cond = 1:n_conditions
        % Generate time vector
        t = (0:n_samples-1) / fs;
        
        % Generate base EEG signal (mixture of frequency bands)
        % Alpha component (main signal of interest)
        alpha_component = generateBandLimitedNoise(alpha_freq, fs, n_samples) * 3;
        
        % Other frequency components
        beta_component = generateBandLimitedNoise(beta_freq, fs, n_samples) * 2;
        theta_component = generateBandLimitedNoise(theta_freq, fs, n_samples) * 2.5;
        delta_component = generateBandLimitedNoise(delta_freq, fs, n_samples) * 4;
        
        % Combine EEG components
        clean_eeg = alpha_component + beta_component + theta_component + delta_component;
        
        % Add eye blink artifacts
        eye_artifacts = generateEyeArtifacts(t, eye_blink_freq, ...
            eye_artifact_amplitude(cond) * subject_eye_tendency, fs);
        
        % Add line noise (50/60 Hz)
        line_noise = 0.5 * sin(2*pi*50*t) + 0.3 * sin(2*pi*60*t);
        
        % Add muscle artifacts (high frequency)
        muscle_noise = 0.8 * generateBandLimitedNoise([50 100], fs, n_samples);
        
        % Combine all components
        raw_eeg = clean_eeg + eye_artifacts + line_noise + muscle_noise;
        
        % Add white noise
        raw_eeg = raw_eeg + 0.5 * randn(size(raw_eeg));
        
        % Store raw data
        raw_eeg_data{subj, cond} = raw_eeg;
        
        % Extract alpha power using Welch's method
        [psd, freqs] = pwelch(raw_eeg, hamming(fs), fs/2, fs, fs);
        alpha_idx = freqs >= alpha_freq(1) & freqs <= alpha_freq(2);
        alpha_power_raw = mean(psd(alpha_idx));
        alpha_power_db = 10 * log10(alpha_power_raw);
        
        % Apply treatment effect
        treatment_effect_subj = treatment_effect(cond) + 0.5 * randn; % Add noise
        alpha_power_db = alpha_power_db + treatment_effect_subj;
        
        % Store processed alpha power
        eeg_alpha_power(subj, cond) = alpha_power_db;
    end
end

%% Display data characteristics
fprintf('\n=== DATA CHARACTERISTICS ===\n');
condition_names = {'Pre-Treatment', 'During Treatment', 'Post-Treatment (24mo)'};

for cond = 1:n_conditions
    fprintf('\n%s:\n', condition_names{cond});
    fprintf('  Mean Alpha Power: %.2f ± %.2f dB\n', ...
        mean(eeg_alpha_power(:, cond)), std(eeg_alpha_power(:, cond)));
    fprintf('  Range: [%.2f, %.2f] dB\n', ...
        min(eeg_alpha_power(:, cond)), max(eeg_alpha_power(:, cond)));
end

%% Visualize raw EEG examples
figure('Name', 'Sample Raw EEG Data', 'Position', [50 50 1200 800]);

% Show 3 example subjects
example_subjects = [1, 10, 20];
for i = 1:length(example_subjects)
    subj = example_subjects(i);
    for cond = 1:n_conditions
        subplot(length(example_subjects), n_conditions, (i-1)*n_conditions + cond);
        
        t_plot = (0:length(raw_eeg_data{subj, cond})-1) / fs;
        plot(t_plot, raw_eeg_data{subj, cond});
        
        title(sprintf('Subject %d - %s', subj, condition_names{cond}));
        xlabel('Time (s)');
        ylabel('Amplitude (?V)');
        ylim([-30 30]);
        grid on;
    end
end

%% Create power spectral density plots
figure('Name', 'Power Spectral Density Analysis', 'Position', [100 100 1000 600]);

colors = {'r', 'g', 'b'};
for cond = 1:n_conditions
    % Calculate group average PSD
    all_psds = [];
    for subj = 1:n_subjects
        [psd, freqs] = pwelch(raw_eeg_data{subj, cond}, hamming(fs), fs/2, fs, fs);
        all_psds(:, subj) = 10*log10(psd);
    end
    
    mean_psd = mean(all_psds, 2);
    std_psd = std(all_psds, 0, 2);
    
    % Plot with error bars
    subplot(1,2,1);
    hold on;
    plot(freqs, mean_psd, 'Color', colors{cond}, 'LineWidth', 2, ...
        'DisplayName', condition_names{cond});
    
    % Highlight alpha band
    alpha_idx = freqs >= 8 & freqs <= 13;
    subplot(1,2,2);
    hold on;
    errorbar(cond, mean(mean_psd(alpha_idx)), std(mean_psd(alpha_idx)), ...
        'o', 'Color', colors{cond}, 'MarkerSize', 8, 'LineWidth', 2, ...
        'MarkerFaceColor', colors{cond});
end

subplot(1,2,1);
xlabel('Frequency (Hz)');
ylabel('Power (dB/Hz)');
title('Group Average Power Spectral Density');
xlim([0 50]);
legend('show');
grid on;

subplot(1,2,2);
xlabel('Treatment Phase');
ylabel('Alpha Power (dB)');
title('Alpha Band Power Across Conditions');
xticks(1:3);
xticklabels({'Pre', 'During', 'Post'});
grid on;

%% Run Repeated Measures ANOVA
fprintf('\n=== RUNNING REPEATED MEASURES ANOVA ===\n');
fprintf('Analyzing alpha power changes across treatment phases...\n');

% Run the custom ANOVA function
[results, stats] = repeatedMeasuresANOVA(eeg_alpha_power, 0.05);

%% Additional EEG-specific analyses
fprintf('\n=== ADDITIONAL EEG ANALYSES ===\n');

% Pairwise t-tests (post-hoc analysis)
fprintf('\nPairwise Comparisons (paired t-tests):\n');
comparisons = {[1,2], [1,3], [2,3]};
comparison_names = {'Pre vs During', 'Pre vs Post', 'During vs Post'};

for i = 1:length(comparisons)
    idx1 = comparisons{i}(1);
    idx2 = comparisons{i}(2);
    
    [h, p, ci, stats_t] = ttest(eeg_alpha_power(:, idx1), eeg_alpha_power(:, idx2));
    
    % Calculate Cohen's d
    mean_diff = mean(eeg_alpha_power(:, idx1) - eeg_alpha_power(:, idx2));
    pooled_std = sqrt((std(eeg_alpha_power(:, idx1))^2 + std(eeg_alpha_power(:, idx2))^2) / 2);
    cohens_d = mean_diff / pooled_std;
    
    fprintf('%s: t(%d) = %.3f, p = %.4f, d = %.3f\n', ...
        comparison_names{i}, stats_t.df, stats_t.tstat, p, cohens_d);
    
    if p < 0.05/3 % Bonferroni correction
        fprintf('  -> SIGNIFICANT (Bonferroni corrected)\n');
    else
        fprintf('  -> Not significant (Bonferroni corrected)\n');
    end
end

% Effect size interpretation
fprintf('\nClinical Interpretation:\n');
pre_mean = mean(eeg_alpha_power(:, 1));
post_mean = mean(eeg_alpha_power(:, 3));
percent_change = ((post_mean - pre_mean) / pre_mean) * 100;

fprintf('Alpha power changed by %.1f%% from pre- to post-treatment\n', percent_change);

% Count responders (subjects showing >10% improvement)
responders = sum((eeg_alpha_power(:, 3) - eeg_alpha_power(:, 1)) ./ eeg_alpha_power(:, 1) < -0.10);
fprintf('%d out of %d subjects (%.1f%%) showed >10%% improvement\n', ...
    responders, n_subjects, (responders/n_subjects)*100);

%% Create comprehensive visualization
figure('Name', 'EEG Treatment Response Analysis', 'Position', [150 50 1400 900]);

% Individual trajectories
subplot(2,3,1);
plot(1:3, eeg_alpha_power', 'o-', 'LineWidth', 1, 'MarkerSize', 4);
hold on;
plot(1:3, mean(eeg_alpha_power), 'ro-', 'LineWidth', 3, 'MarkerSize', 10);
title('Individual Response Trajectories');
xlabel('Treatment Phase');
ylabel('Alpha Power (dB)');
xticks(1:3);
xticklabels({'Pre', 'During', 'Post'});
grid on;

% Box plots
subplot(2,3,2);
boxplot(eeg_alpha_power, 'Labels', {'Pre', 'During', 'Post'});
title('Alpha Power Distribution');
ylabel('Power (dB)');
grid on;

% Change from baseline
subplot(2,3,3);
changes = eeg_alpha_power - repmat(eeg_alpha_power(:,1), 1, 3);
boxplot(changes(:,2:3), 'Labels', {'During-Pre', 'Post-Pre'});
title('Change from Baseline');
ylabel('Change in Alpha Power (dB)');
yline(0, 'r--', 'LineWidth', 1);
grid on;

% Correlation matrix
subplot(2,3,4);
corr_matrix = corr(eeg_alpha_power);
imagesc(corr_matrix);
colorbar;
title('Correlation Matrix');
xticks(1:3);
yticks(1:3);
xticklabels({'Pre', 'During', 'Post'});
yticklabels({'Pre', 'During', 'Post'});

% Histogram of changes
subplot(2,3,5);
histogram(eeg_alpha_power(:,3) - eeg_alpha_power(:,1), 10);
title('Distribution of Pre-Post Changes');
xlabel('Change in Alpha Power (dB)');
ylabel('Number of Subjects');
xline(0, 'r--', 'LineWidth', 2);
grid on;

% Time course with confidence intervals
subplot(2,3,6);
means = mean(eeg_alpha_power);
sems = std(eeg_alpha_power) / sqrt(n_subjects);
errorbar(1:3, means, sems, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
title('Group Mean ± SEM');
xlabel('Treatment Phase');
ylabel('Alpha Power (dB)');
xticks(1:3);
xticklabels({'Pre', 'During', 'Post'});
grid on;

%% Save all plots from test session
try
    fprintf('\n=== SAVING ALL PLOTS ===\n');
    
    % Get all figure handles
    fig_handles = findobj('Type', 'figure');
    fig_count = length(fig_handles);
    
    % Save each figure with descriptive names
    plot_names = {
        'EEG_Raw_Data_Examples',
        'EEG_Power_Spectral_Density', 
        'ANOVA_Main_Results',
        'ANOVA_Outlier_Detection',
        'EEG_Treatment_Response_Analysis'
    };
    
    for i = 1:min(fig_count, length(plot_names))
        figure(i);
        
        % Save as both PNG and FIG
        png_name = [plot_names{i}, '.png'];
        fig_name = [plot_names{i}, '.fig'];
        
        saveas(gcf, png_name);
        saveas(gcf, fig_name);
        
        fprintf('  ? Saved: %s\n', png_name);
        fprintf('  ? Saved: %s\n', fig_name);
    end
    
    fprintf('\nAll plots saved successfully to current directory!\n');
    fprintf('Total files saved: %d PNG + %d FIG = %d files\n', ...
        min(fig_count, length(plot_names)), min(fig_count, length(plot_names)), ...
        2 * min(fig_count, length(plot_names)));
    
catch ME
    fprintf('\nWarning: Could not save plots.\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Make sure you have write permissions in current directory.\n');
end

fprintf('\n=== TEST COMPLETE ===\n');
fprintf('Generated realistic EEG data with:\n');
fprintf('- Eye blink artifacts (decreasing with treatment compliance)\n');
fprintf('- Line noise (50/60 Hz)\n');
fprintf('- Muscle artifacts\n');
fprintf('- Multiple frequency bands\n');
fprintf('- Realistic treatment response pattern\n');

end

%% Helper functions
function noise = generateBandLimitedNoise(freq_range, fs, n_samples)
% Generate band-limited white noise
    % Create frequency domain representation
    freqs = linspace(0, fs/2, floor(n_samples/2) + 1);
    
    % Create filter
    filter_response = zeros(size(freqs));
    filter_idx = freqs >= freq_range(1) & freqs <= freq_range(2);
    filter_response(filter_idx) = 1;
    
    % Generate white noise in frequency domain
    white_noise_fft = randn(1, floor(n_samples/2) + 1) + 1i * randn(1, floor(n_samples/2) + 1);
    white_noise_fft(1) = real(white_noise_fft(1)); % DC component
    if mod(n_samples, 2) == 0
        white_noise_fft(end) = real(white_noise_fft(end)); % Nyquist frequency
    end
    
    % Apply filter
    filtered_fft = white_noise_fft .* filter_response;
    
    % Convert back to time domain
    if mod(n_samples, 2) == 0
        full_fft = [filtered_fft, conj(filtered_fft(end-1:-1:2))];
    else
        full_fft = [filtered_fft, conj(filtered_fft(end:-1:2))];
    end
    
    noise = real(ifft(full_fft));
    noise = noise(1:n_samples);
end

function artifacts = generateEyeArtifacts(t, blink_rate, amplitude, fs)
% Generate realistic eye blink artifacts
    n_samples = length(t);
    artifacts = zeros(1, n_samples);
    
    % Generate random blink times
    avg_interval = 1 / blink_rate;
    blink_times = [];
    current_time = avg_interval * rand; % Random start
    
    while current_time < t(end)
        blink_times = [blink_times, current_time];
        % Next blink with some variability
        current_time = current_time + avg_interval * (0.5 + rand);
    end
    
    % Add blink artifacts
    for blink_time = blink_times
        [~, blink_idx] = min(abs(t - blink_time));
        
        % Create blink shape (asymmetric, sharp rise, slower fall)
        blink_duration = 0.3; % 300ms blink
        blink_samples = round(blink_duration * fs);
        blink_start = max(1, blink_idx - round(blink_samples/4));
        blink_end = min(n_samples, blink_idx + round(3*blink_samples/4));
        
        blink_indices = blink_start:blink_end;
        
        % Create asymmetric blink waveform
        t_blink = (blink_indices - blink_idx) / fs;
        
        % Double exponential for realistic blink shape
        blink_shape = amplitude * exp(-abs(t_blink)/0.05) .* (1 - exp(-abs(t_blink)/0.01));
        
        % Add some randomness to blink amplitude
        blink_shape = blink_shape * (0.7 + 0.6 * rand);
        
        artifacts(blink_indices) = artifacts(blink_indices) + blink_shape;
    end
    
    % Add slower eye movements (saccades and drifts)
    eye_movement_freq = 0.1; % Very low frequency drifts
    eye_movements = amplitude * 0.3 * sin(2*pi*eye_movement_freq*t + 2*pi*rand) .* ...
                   (1 + 0.5*randn(size(t)));
    
    artifacts = artifacts + eye_movements;
end