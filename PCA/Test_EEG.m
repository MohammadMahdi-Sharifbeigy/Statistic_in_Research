%% EEG PCA Analysis Test Script
clear all; close all; clc;

%% Parameters for EEG Data Generation
fs = 250;           % Sampling frequency (Hz)
duration = 60;      % Duration in seconds
t = 0:1/fs:duration-1/fs;
n_samples = length(t);
n_channels = 8;

% Channel names (standard 10-20 system subset)
channel_names = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'};

fprintf('=== Generating Synthetic 8-Channel EEG Data ===\n');
fprintf('Sampling Rate: %d Hz\n', fs);
fprintf('Duration: %d seconds\n', duration);
fprintf('Number of samples: %d\n', n_samples);
fprintf('Number of channels: %d\n', n_channels);

%% 1. Generate Base EEG Signals
fprintf('\n1. Generating base EEG rhythms...\n');

% Initialize EEG data matrix
eeg_data = zeros(n_samples, n_channels);

% Base brain rhythms with different amplitudes per channel
% Delta (1-4 Hz)
delta_freq = [1.5, 2, 2.5, 1.8, 2.2, 1.9, 2.8, 2.1];
delta_amp = [8, 7, 9, 8.5, 7.5, 8.2, 9.1, 7.8];

% Theta (4-8 Hz)  
theta_freq = [5, 6, 5.5, 6.2, 5.8, 6.5, 5.3, 6.8];
theta_amp = [12, 11, 13, 12.5, 11.8, 12.2, 13.5, 11.5];

% Alpha (8-12 Hz) - stronger in posterior channels
alpha_freq = [9, 9.5, 10, 10.2, 9.8, 10.5, 11, 10.8];
alpha_amp = [15, 14, 16, 15.5, 18, 19, 22, 21]; % Stronger in P3, P4

% Beta (12-30 Hz) - stronger in frontal/central channels
beta_freq = [18, 19, 17, 20, 22, 21, 16, 18];
beta_amp = [25, 24, 23, 26, 28, 27, 20, 19]; % Stronger in frontal/central

% Generate base signals for each channel
for ch = 1:n_channels
    % Combine different frequency bands
    eeg_data(:, ch) = ...
        delta_amp(ch) * sin(2*pi*delta_freq(ch)*t + rand*2*pi) + ...
        theta_amp(ch) * sin(2*pi*theta_freq(ch)*t + rand*2*pi) + ...
        alpha_amp(ch) * sin(2*pi*alpha_freq(ch)*t + rand*2*pi) + ...
        beta_amp(ch) * sin(2*pi*beta_freq(ch)*t + rand*2*pi);
    
    % Add some phase coupling between adjacent channels
    if ch > 1
        coupling_strength = 0.3;
        eeg_data(:, ch) = eeg_data(:, ch) + coupling_strength * eeg_data(:, ch-1);
    end
end

%% 2. Add Realistic Noise and Artifacts
fprintf('2. Adding noise and artifacts...\n');

% 2.1 Gaussian noise (background EEG activity)
noise_level = 5; % microvolts
gaussian_noise = noise_level * randn(size(eeg_data));
eeg_data = eeg_data + gaussian_noise;

% 2.2 50Hz Power line noise (common in EEG)
powerline_freq = 50; % Hz
powerline_amp = 3; % microvolts
for ch = 1:n_channels
    powerline_phase = rand * 2 * pi; % Random phase per channel
    powerline_noise = powerline_amp * sin(2*pi*powerline_freq*t + powerline_phase);
    eeg_data(:, ch) = eeg_data(:, ch) + powerline_noise';
end

% 2.3 Eye blink artifacts (mainly in frontal channels Fp1, Fp2)
fprintf('   - Adding eye blink artifacts...\n');
n_blinks = 20; % Number of eye blinks
blink_times = sort(rand(n_blinks, 1) * duration); % Random blink times
blink_amplitude = [80, 75, 30, 25, 10, 8, 5, 5]; % Amplitude per channel (stronger in frontal)

for i = 1:n_blinks
    blink_start = round(blink_times(i) * fs);
    blink_duration = round((0.2 + 0.1*rand) * fs); % 200-300ms duration
    
    if blink_start + blink_duration <= n_samples
        % Create blink waveform (exponential decay)
        blink_samples = 1:blink_duration;
        blink_shape = exp(-blink_samples/(blink_duration/3)); % Exponential decay
        
        for ch = 1:n_channels
            blink_artifact = blink_amplitude(ch) * blink_shape;
            eeg_data(blink_start:blink_start+blink_duration-1, ch) = ...
                eeg_data(blink_start:blink_start+blink_duration-1, ch) + blink_artifact';
        end
    end
end

% 2.4 Muscle artifacts (EMG) - random bursts
fprintf('   - Adding muscle artifacts...\n');
n_muscle_artifacts = 8;
muscle_times = sort(rand(n_muscle_artifacts, 1) * duration);

for i = 1:n_muscle_artifacts
    muscle_start = round(muscle_times(i) * fs);
    muscle_duration = round((0.5 + 0.5*rand) * fs); % 0.5-1s duration
    
    if muscle_start + muscle_duration <= n_samples
        % High frequency muscle activity (30-100 Hz)
        muscle_freq = 30 + 70*rand; % Random frequency between 30-100 Hz
        muscle_samples = muscle_start:muscle_start+muscle_duration-1;
        muscle_t = muscle_samples / fs;
        
        % Random channels affected (usually temporal/frontal)
        affected_channels = randsample(n_channels, round(n_channels/2));
        
        for ch = affected_channels'
            muscle_amplitude = 20 + 15*rand; % 20-35 microvolts
            muscle_artifact = muscle_amplitude * sin(2*pi*muscle_freq*muscle_t);
            % Add envelope to make it realistic
            envelope = exp(-abs(muscle_t - mean(muscle_t))*5);
            eeg_data(muscle_samples, ch) = eeg_data(muscle_samples, ch) + (muscle_artifact .* envelope)';
        end
    end
end

% 2.5 Electrode artifacts (sudden jumps/drifts)
fprintf('   - Adding electrode artifacts...\n');
n_electrode_artifacts = 3;
for i = 1:n_electrode_artifacts
    artifact_channel = randi(n_channels);
    artifact_time = round(rand * n_samples);
    artifact_duration = round((2 + 3*rand) * fs); % 2-5 seconds
    
    if artifact_time + artifact_duration <= n_samples
        % Slow drift
        drift_amplitude = 30 + 20*rand;
        drift_samples = artifact_time:artifact_time+artifact_duration-1;
        drift_pattern = drift_amplitude * (1 - exp(-(drift_samples-artifact_time)/(artifact_duration/3)));
        eeg_data(drift_samples, artifact_channel) = eeg_data(drift_samples, artifact_channel) + drift_pattern';
    end
end

fprintf('EEG data generation complete!\n');

%% 3. Apply PCA Analysis
fprintf('\n=== Applying PCA Analysis ===\n');

% Create PCA object
pca_eig = CustomPCA(n_channels);

% Fit PCA model
fprintf('Fitting PCA...\n');
pca_eig = pca_eig.fit(eeg_data);

% Transform data
eeg_pca = pca_eig.transform(eeg_data);

% Display summary
pca_eig.summary();

%% 4. Create Comprehensive Plots
fprintf('\n=== Creating Plots ===\n');

% Set up figure properties
set(0, 'DefaultFigureRenderer', 'painters'); % Better for saving
fig_width = 1200;
fig_height = 800;

%% Plot 1: Original EEG Data
figure('Name', 'Original EEG Data', 'Position', [100, 100, fig_width, fig_height]);

% Time vector for plotting (in seconds)
t_plot = (0:n_samples-1) / fs;

% Plot first 10 seconds for clarity
plot_duration = 10; % seconds
plot_samples = 1:plot_duration*fs;

subplot(2,1,1);
% Plot all channels with offset for visibility
channel_offset = 100; % microvolts
colors = lines(n_channels);

hold on;
for ch = 1:n_channels
    plot(t_plot(plot_samples), eeg_data(plot_samples, ch) + (ch-1)*channel_offset, ...
         'Color', colors(ch,:), 'LineWidth', 1);
end

% Customize plot
xlabel('Time (s)');
ylabel('Amplitude (?V)');
title('Original 8-Channel EEG Data (First 10 seconds)');
grid on;

% Add channel labels
yticks((0:n_channels-1)*channel_offset);
yticklabels(channel_names);
ylim([-channel_offset/2, (n_channels-0.5)*channel_offset]);
xlim([0, plot_duration]);

% Add legend for artifact identification
text(0.5, (n_channels-0.2)*channel_offset, 'Artifacts present: Eye blinks, Muscle activity, Electrode drift, 50Hz noise', ...
     'BackgroundColor', 'white', 'EdgeColor', 'black');

% Power spectral density
subplot(2,1,2);
% Compute PSD for each channel
f_psd = 0:fs/n_samples:fs/2;
psd_data = zeros(length(f_psd), n_channels);

for ch = 1:n_channels
    [pxx, f] = periodogram(eeg_data(:,ch), [], [], fs);
    psd_data(:, ch) = 10*log10(pxx(1:length(f_psd)));
end

% Plot PSD
for ch = 1:n_channels
    semilogy(f_psd, 10.^(psd_data(:,ch)/10), 'Color', colors(ch,:), 'LineWidth', 1.5);
    hold on;
end

xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (?V²/Hz)');
title('Power Spectral Density - All Channels');
legend(channel_names, 'Location', 'northeast');
grid on;
xlim([0, 50]); % Focus on 0-50 Hz range

% Mark typical EEG bands
xline(4, '--', 'Delta|Theta', 'LabelVerticalAlignment', 'bottom');
xline(8, '--', 'Theta|Alpha', 'LabelVerticalAlignment', 'bottom');
xline(12, '--', 'Alpha|Beta', 'LabelVerticalAlignment', 'bottom');
xline(30, '--', 'Beta|Gamma', 'LabelVerticalAlignment', 'bottom');

% Save plot
saveas(gcf, 'EEG_Original_Data.png');
saveas(gcf, 'EEG_Original_Data.fig');

%% Plot 2: PCA Results
figure('Name', 'PCA Analysis Results', 'Position', [200, 50, fig_width, fig_height]);

% Explained variance
subplot(2,3,1);
bar(pca_eig.explained_variance_ratio_);
xlabel('Principal Component');
ylabel('Explained Variance Ratio');
title('Explained Variance Ratio');
grid on;

% Cumulative explained variance
subplot(2,3,2);
plot(1:n_channels, cumsum(pca_eig.explained_variance_ratio_), 'o-', 'LineWidth', 2);
xlabel('Number of Components');
ylabel('Cumulative Explained Variance');
title('Cumulative Explained Variance');
grid on;
ylim([0, 1]);

% Eigenvalues (scree plot)
subplot(2,3,3);
semilogy(1:length(pca_eig.explained_variance_), pca_eig.explained_variance_, 'o-', 'LineWidth', 2);
xlabel('Component');
ylabel('Eigenvalue');
title('Scree Plot (Eigenvalues)');
grid on;

% First 4 Principal Components (time series)
subplot(2,3,4);
for pc = 1:4
    plot(t_plot(plot_samples), eeg_pca(plot_samples, pc) + (pc-1)*50, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (s)');
ylabel('PC Amplitude');
title('First 4 Principal Components');
yticks((0:3)*50);
yticklabels({'PC1', 'PC2', 'PC3', 'PC4'});
grid on;
xlim([0, plot_duration]);

% Component loadings (spatial patterns)
subplot(2,3,5);
imagesc(pca_eig.components_');
colormap('hot');
colorbar;
xlabel('Principal Component');
ylabel('EEG Channel');
title('Component Loadings (Spatial Patterns)');
yticks(1:n_channels);
yticklabels(channel_names);
xticks(1:n_channels);

% Reconstruction error vs number of components
subplot(2,3,6);
n_comp_test = 1:n_channels;
recon_error = zeros(size(n_comp_test));

for i = 1:length(n_comp_test)
    pca_test = CustomPCA(n_comp_test(i));
    pca_test = pca_test.fit(eeg_data);
    recon_error(i) = pca_test.reconstruction_error(eeg_data);
end

semilogy(n_comp_test, recon_error, 'o-', 'LineWidth', 2);
xlabel('Number of Components');
ylabel('Reconstruction Error (MSE)');
title('Reconstruction Error vs Components');
grid on;

% Save plot
saveas(gcf, 'EEG_PCA_Analysis.png');
saveas(gcf, 'EEG_PCA_Analysis.fig');

%% Plot 3: Feature Importance Analysis
figure('Name', 'Feature Importance Analysis', 'Position', [300, 0, fig_width, fig_height]);

% Calculate different types of feature importance
fprintf('Computing feature importance metrics...\n');

% 1. Variance-based importance
channel_variance = var(eeg_data, 0, 1);
variance_importance = channel_variance / sum(channel_variance);

% 2. Component loading-based importance
loading_importance = sum(pca_eig.components_.^2, 2)';
loading_importance = loading_importance / sum(loading_importance);

% 3. Weighted loading importance
weighted_loadings = zeros(1, n_channels);
for ch = 1:n_channels
    for comp = 1:pca_eig.n_components
        weighted_loadings(ch) = weighted_loadings(ch) + ...
            (pca_eig.components_(ch, comp)^2) * pca_eig.explained_variance_ratio_(comp);
    end
end
weighted_importance = weighted_loadings / sum(weighted_loadings);

% Feature importance comparison
subplot(2,2,1);
x_pos = 1:n_channels;
bar_width = 0.25;

bar(x_pos - bar_width, variance_importance, bar_width, 'DisplayName', 'Variance-based');
hold on;
bar(x_pos, loading_importance, bar_width, 'DisplayName', 'Loading-based');
bar(x_pos + bar_width, weighted_importance, bar_width, 'DisplayName', 'Weighted Loading');

xlabel('EEG Channel');
ylabel('Importance Score');
title('Feature Importance Comparison');
legend('Location', 'northeast');
xticks(x_pos);
xticklabels(channel_names);
grid on;
xtickangle(45);

% Component-wise channel contributions
subplot(2,2,2);
comp_contributions = abs(pca_eig.components_(:, 1:4));
imagesc(comp_contributions');
colormap('hot');
colorbar;
xlabel('EEG Channel');
ylabel('Principal Component');
title('Channel Contributions to Components');
xticks(1:n_channels);
xticklabels(channel_names);
yticks(1:4);
yticklabels({'PC1', 'PC2', 'PC3', 'PC4'});
xtickangle(45);

% Cumulative feature importance
subplot(2,2,3);
[~, sort_idx] = sort(weighted_importance, 'descend');
cumulative_importance = cumsum(weighted_importance(sort_idx));

bar(cumulative_importance);
hold on;
yline(0.8, 'r--', 'LineWidth', 2, 'Label', '80% Threshold');
yline(0.95, 'g--', 'LineWidth', 2, 'Label', '95% Threshold');

xlabel('Channel (Sorted by Importance)');
ylabel('Cumulative Importance');
title('Cumulative Feature Importance');
xticks(1:n_channels);
xticklabels(channel_names(sort_idx));
legend('Location', 'southeast');
grid on;
xtickangle(45);

% Find channels needed for thresholds
channels_80 = find(cumulative_importance >= 0.8, 1, 'first');
channels_95 = find(cumulative_importance >= 0.95, 1, 'first');

text(0.6*n_channels, 0.3, sprintf('80%% captured by %d channels\n95%% captured by %d channels', ...
     channels_80, channels_95), 'BackgroundColor', 'white', 'EdgeColor', 'black');

% Frequency band importance per channel
subplot(2,2,4);
bands = {'Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-12Hz)', 'Beta (12-30Hz)', 'Gamma (30-50Hz)'};
band_ranges = [1 4; 4 8; 8 12; 12 30; 30 50];
band_power = zeros(n_channels, length(bands));

for ch = 1:n_channels
    [pxx, f] = periodogram(eeg_data(:, ch), [], [], fs);
    
    for band = 1:length(bands)
        freq_idx = f >= band_ranges(band, 1) & f <= band_ranges(band, 2);
        band_power(ch, band) = sum(pxx(freq_idx));
    end
end

% Normalize by total power per channel
band_power_norm = band_power ./ sum(band_power, 2);

imagesc(band_power_norm');
colormap('parula');
colorbar;
xlabel('EEG Channel');
ylabel('Frequency Band');
title('Relative Power Distribution');
xticks(1:n_channels);
xticklabels(channel_names);
yticks(1:length(bands));
yticklabels(bands);
xtickangle(45);

% Save plot
saveas(gcf, 'EEG_Feature_Importance.png');
saveas(gcf, 'EEG_Feature_Importance.fig');

%% Plot 4: 2D PCA Projection
figure('Name', 'PCA Projection of EEG Data', 'Position', [400, 100, fig_width, 600]);

% Create data categories based on EEG characteristics
fprintf('Categorizing EEG data for 2D projection...\n');

sample_features = zeros(n_samples, 4);

for i = 1:n_samples
    sample = eeg_data(i, :);
    
    % Feature 1: High-frequency content (muscle artifacts)
    sample_features(i, 1) = sum(abs(sample - mean(sample)) > 2*std(sample));
    
    % Feature 2: Frontal dominance (eye artifacts)
    frontal_power = sum(sample([1,2]).^2); % Fp1, Fp2
    total_power = sum(sample.^2);
    sample_features(i, 2) = frontal_power / (total_power + eps);
    
    % Feature 3: Posterior alpha activity
    posterior_power = sum(sample([7,8]).^2); % P3, P4
    sample_features(i, 3) = posterior_power / (total_power + eps);
    
    % Feature 4: Overall amplitude
    sample_features(i, 4) = std(sample);
end

% Define categories
categories = cell(n_samples, 1);

% Normal EEG
normal_idx = sample_features(:,1) < 2 & sample_features(:,2) < 0.4 & sample_features(:,4) < 50;
categories(normal_idx) = {'NORMAL'};

% Eye artifacts
eye_idx = sample_features(:,2) > 0.4 & ~normal_idx;
categories(eye_idx) = {'EYE_ARTIFACT'};

% Muscle artifacts
muscle_idx = sample_features(:,1) >= 2 & sample_features(:,4) > 40 & ~eye_idx;
categories(muscle_idx) = {'MUSCLE_ARTIFACT'};

% High amplitude events
artifact_idx = sample_features(:,4) > 70 & ~normal_idx & ~eye_idx & ~muscle_idx;
categories(artifact_idx) = {'ELECTRODE_ARTIFACT'};

% Remaining samples
remaining_idx = cellfun(@isempty, categories);
categories(remaining_idx) = {'OTHER'};

% Plot categories
unique_categories = {'NORMAL', 'EYE_ARTIFACT', 'MUSCLE_ARTIFACT', 'ELECTRODE_ARTIFACT', 'OTHER'};
category_colors = [0.2, 0.8, 0.2; 1, 0.5, 0; 0.8, 0.2, 0.2; 0.6, 0.2, 0.8; 0.5, 0.5, 0.5];

pc1_data = eeg_pca(:, 1);
pc2_data = eeg_pca(:, 2);

hold on;
legend_handles = [];
legend_labels = {};

for i = 1:length(unique_categories)
    cat_idx = strcmp(categories, unique_categories{i});
    if sum(cat_idx) > 0
        h = scatter(pc1_data(cat_idx), pc2_data(cat_idx), 30, ...
                   category_colors(i, :), 'filled', ...
                   'MarkerEdgeColor', 'black', 'LineWidth', 0.5);
        legend_handles = [legend_handles, h];
        legend_labels{end+1} = sprintf('%s (%d)', strrep(unique_categories{i}, '_', ' '), sum(cat_idx));
    end
end

xlabel(sprintf('PC1 (%.2f%% variance)', pca_eig.explained_variance_ratio_(1)*100), 'FontSize', 12);
ylabel(sprintf('PC2 (%.2f%% variance)', pca_eig.explained_variance_ratio_(2)*100), 'FontSize', 12);
title('PCA Projection of EEG Data Types', 'FontSize', 14, 'FontWeight', 'bold');

legend(legend_handles, legend_labels, 'Location', 'eastoutside', 'FontSize', 10);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3);

% Statistics text box
total_var_2d = sum(pca_eig.explained_variance_ratio_(1:2)) * 100;
text_str = {
    'Statistics:';
    sprintf('Total variance: %.1f%%', total_var_2d);
    sprintf('PC1: %.2f%%, PC2: %.2f%%', ...
            pca_eig.explained_variance_ratio_(1)*100, ...
            pca_eig.explained_variance_ratio_(2)*100);
    sprintf('Samples: %d', n_samples);
};

xlims = xlim;
ylims = ylim;
text(xlims(1) + 0.02*(xlims(2)-xlims(1)), ylims(2) - 0.02*(ylims(2)-ylims(1)), ...
     text_str, 'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'FontSize', 9, 'Margin', 5);

% Save plot
saveas(gcf, 'EEG_PCA_Projection.png');
saveas(gcf, 'EEG_PCA_Projection.fig');

%% 5. Save Results and Summary
fprintf('\n=== Saving Results ===\n');

% Save workspace
save('EEG_PCA_Analysis_Results.mat', 'eeg_data', 'pca_eig', 'eeg_pca', ...
     'channel_names', 'fs', 't_plot', 'variance_importance', 'loading_importance', ...
     'weighted_importance', 'band_power', 'channels_80', 'channels_95', ...
     'categories', 'sample_features');

% Create summary report
fid = fopen('EEG_PCA_Analysis_Report.txt', 'w');
fprintf(fid, '=== EEG PCA Analysis Report ===\n');
fprintf(fid, 'Generated on: %s\n\n', datestr(now));

fprintf(fid, 'Dataset Characteristics:\n');
fprintf(fid, '- Channels: %d (%s)\n', n_channels, strjoin(channel_names, ', '));
fprintf(fid, '- Sampling Rate: %d Hz\n', fs);
fprintf(fid, '- Duration: %d seconds\n', duration);
fprintf(fid, '- Total Samples: %d\n', n_samples);
fprintf(fid, '- Artifacts Added: Eye blinks, Muscle activity, Electrode drift, Power line noise\n\n');

fprintf(fid, 'PCA Results:\n');
fprintf(fid, '- Total Variance Explained: %.2f%%\n', sum(pca_eig.explained_variance_ratio_)*100);
fprintf(fid, '- First 4 Components Explain: %.2f%%\n', sum(pca_eig.explained_variance_ratio_(1:4))*100);
fprintf(fid, '- Component-wise explained variance:\n');
for i = 1:n_channels
    fprintf(fid, '  PC%d: %.2f%%\n', i, pca_eig.explained_variance_ratio_(i)*100);
end

fprintf(fid, '\nFeature Importance Analysis:\n');
fprintf(fid, '- Most important channels (weighted loading method):\n');
[~, top_idx] = sort(weighted_importance, 'descend');
for i = 1:min(5, n_channels)
    fprintf(fid, '  %d. %s: %.3f (%.1f%%)\n', i, channel_names{top_idx(i)}, ...
            weighted_importance(top_idx(i)), weighted_importance(top_idx(i))*100);
end

fprintf(fid, '- Channels needed for variance thresholds:\n');
fprintf(fid, '  80%% importance: %d channels (%s)\n', channels_80, ...
        strjoin(channel_names(sort_idx(1:channels_80)), ', '));
fprintf(fid, '  95%% importance: %d channels (%s)\n', channels_95, ...
        strjoin(channel_names(sort_idx(1:channels_95)), ', '));

fprintf(fid, '\nData Categories (2D Projection):\n');
for i = 1:length(unique_categories)
    count = sum(strcmp(categories, unique_categories{i}));
    fprintf(fid, '- %s: %d samples (%.1f%%)\n', unique_categories{i}, count, 100*count/n_samples);
end

fprintf(fid, '\nGenerated Files:\n');
fprintf(fid, '- EEG_Original_Data.png/fig\n');
fprintf(fid, '- EEG_PCA_Analysis.png/fig\n');
fprintf(fid, '- EEG_Feature_Importance.png/fig\n');
fprintf(fid, '- EEG_PCA_Projection.png/fig\n');
fprintf(fid, '- EEG_PCA_Analysis_Results.mat\n');
fprintf(fid, '- EEG_PCA_Analysis_Report.txt\n');

fclose(fid);

fprintf('Analysis complete! Files saved:\n');
fprintf('- Plots: EEG_*.png and EEG_*.fig\n');
fprintf('- Data: EEG_PCA_Analysis_Results.mat\n');
fprintf('- Report: EEG_PCA_Analysis_Report.txt\n');

%% 6. Display Results Summary
fprintf('\n=== Analysis Summary ===\n');

% Feature importance summary
fprintf('Top 3 most important channels:\n');
for i = 1:3
    fprintf('%d. %s: %.3f (%.1f%%)\n', i, channel_names{top_idx(i)}, ...
            weighted_importance(top_idx(i)), weighted_importance(top_idx(i))*100);
end

% 2D projection summary
total_var_2d = sum(pca_eig.explained_variance_ratio_(1:2)) * 100;
fprintf('\n2D PCA Projection Results:\n');
fprintf('- First 2 components explain %.1f%% of variance\n', total_var_2d);

for i = 1:length(unique_categories)
    count = sum(strcmp(categories, unique_categories{i}));
    fprintf('- %s: %d samples (%.1f%%)\n', unique_categories{i}, count, 100*count/n_samples);
end

fprintf('\n=== Test Complete ===\n');