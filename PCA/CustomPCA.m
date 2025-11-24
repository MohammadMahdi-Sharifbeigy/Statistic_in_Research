classdef CustomPCA
    % robustness for EEG data analysis.
    % Supports both eigendecomposition and SVD methods.

    properties
        n_components
        components_
        explained_variance_
        explained_variance_ratio_
        mean_
        n_samples_
        n_features_
        noise_variance_  % Added for reconstruction error computation
        svd_solver       % New property to specify decomposition method
        singular_values_ % Store singular values when using SVD
    end

    methods
        function obj = CustomPCA(n_components, varargin)
            % Constructor for the CustomPCA class.
            % n_components: Number of components to keep. Can be an integer
            %               or a float (0 to 1) for cumulative variance.
            % Optional parameters:
            %   'svd_solver': 'auto', 'full', 'eig' (default: 'auto')
            
            if nargin < 1
                n_components = 'all'; % Default case
            end
            obj.n_components = n_components;
            
            % Parse additional parameters
            p = inputParser;
            addParameter(p, 'svd_solver', 'auto', @(x) ismember(x, {'auto', 'full', 'eig'}));
            parse(p, varargin{:});
            
            obj.svd_solver = p.Results.svd_solver;
        end

        function obj = fit(obj, X)
            % Fit the PCA model to the data X.
            
            % Data validation
            obj.validateInput(X);
            
            [obj.n_samples_, obj.n_features_] = size(X);

            % 1. Center the data
            obj.mean_ = mean(X, 1);
            X_centered = X - obj.mean_;

            % 2. Determine which solver to use
            solver_method = obj.chooseSolver(X_centered);
            
            % 3. Apply the chosen decomposition method
            switch solver_method
                case 'svd'
                    obj = obj.fitSVD(X_centered);
                case 'eig'
                    obj = obj.fitEigenDecomposition(X_centered);
            end
            
            % 4. Determine the number of components to keep
            obj = obj.selectComponents();
        end
        
        function obj = fitSVD(obj, X_centered)
            % Fit PCA using Singular Value Decomposition (SVD)
            
            fprintf('Using SVD decomposition method...\n');
            
            % Perform SVD: X_centered = U * S * V'
            [U, S, V] = svd(X_centered, 'econ');
            
            % Extract singular values
            singular_values = diag(S);
            obj.singular_values_ = singular_values;
            
            % Components are the right singular vectors (V)
            % Note: In MATLAB SVD, V contains the right singular vectors as columns
            all_components = V;
            
            % Compute explained variance from singular values
            % For SVD: eigenvalue = (singular_value^2) / (n_samples - 1)
            all_explained_variance = (singular_values.^2) / (obj.n_samples_ - 1);
            
            % Store all results (will be trimmed in selectComponents)
            obj.components_ = all_components;
            obj.explained_variance_ = all_explained_variance;
        end
        
        function obj = fitEigenDecomposition(obj, X_centered)
            % Fit PCA using traditional eigendecomposition
            
            fprintf('Using eigendecomposition method...\n');
            
            % Compute the covariance matrix
            cov_matrix = (X_centered' * X_centered) / (obj.n_samples_ - 1);
            
            % Compute eigenvalues and eigenvectors with numerical stability
            [eigenvectors, D] = eig(cov_matrix);
            eigenvalues = diag(D);
            
            % Handle numerical precision issues
            eigenvalues = real(eigenvalues); % Remove tiny imaginary parts
            eigenvalues = max(eigenvalues, 0); % Ensure non-negative

            % Sort eigenvalues and eigenvectors in descending order
            [eigenvalues, idx] = sort(eigenvalues, 'descend');
            eigenvectors = real(eigenvectors(:, idx)); % Ensure real eigenvectors

            % Store results
            obj.components_ = eigenvectors;
            obj.explained_variance_ = eigenvalues;
            obj.singular_values_ = sqrt(eigenvalues * (obj.n_samples_ - 1));
        end
        
        function solver_method = chooseSolver(obj, X_centered)
            % Choose the appropriate solver based on data characteristics and user preference
            
            [n_samples, n_features] = size(X_centered);
            
            switch obj.svd_solver
                case 'auto'
                    % Heuristic: Use SVD for wide matrices (more features than samples)
                    % or when the number of samples is small
                    if n_features > n_samples || n_samples < 1000
                        solver_method = 'svd';
                    else
                        solver_method = 'eig';
                    end
                    
                case 'full'
                    solver_method = 'svd';
                    
                case 'eig'
                    solver_method = 'eig';
            end
            
            fprintf('Data dimensions: %d samples × %d features\n', n_samples, n_features);
            fprintf('Selected solver: %s\n', solver_method);
        end
        
        function obj = selectComponents(obj)
            % Determine the number of components to keep and trim results accordingly
            
            % Determine the number of components to keep
            if ischar(obj.n_components) && strcmp(obj.n_components, 'all')
                num_components = min(obj.n_samples_, obj.n_features_);
            elseif isnumeric(obj.n_components) && obj.n_components >= 1
                num_components = round(obj.n_components);
            elseif isnumeric(obj.n_components) && obj.n_components < 1 && obj.n_components > 0
                % If n_components is a float, treat it as the desired
                % cumulative explained variance ratio.
                total_variance = sum(obj.explained_variance_);
                cumulative_variance_ratio = cumsum(obj.explained_variance_) / total_variance;
                num_components = find(cumulative_variance_ratio >= obj.n_components, 1, 'first');
                if isempty(num_components)
                    num_components = length(obj.explained_variance_);
                end
            else
                error('Invalid value for n_components. Must be positive integer, float between 0-1, or ''all''.');
            end
            
            % Ensure num_components does not exceed available dimensions
            max_components = min(obj.n_samples_, obj.n_features_);
            max_components = min(max_components, length(obj.explained_variance_));
            num_components = min(num_components, max_components);
            
            if num_components <= 0
                error('Number of components must be positive.');
            end
            
            % Trim to selected number of components
            obj.components_ = obj.components_(:, 1:num_components);
            obj.explained_variance_ = obj.explained_variance_(1:num_components);
            
            if ~isempty(obj.singular_values_)
                obj.singular_values_ = obj.singular_values_(1:num_components);
            end
            
            % Compute explained variance ratio
            total_variance = sum(obj.explained_variance_);
            all_eigenvalues = obj.explained_variance_;
            if length(all_eigenvalues) < length(obj.explained_variance_)
                % We need to get the total variance from all components
                if strcmp(obj.svd_solver, 'svd') || strcmp(obj.chooseSolver(zeros(obj.n_samples_, obj.n_features_)), 'svd')
                    total_variance = sum((obj.singular_values_.^2)) / (obj.n_samples_ - 1);
                else
                    total_variance = sum(obj.explained_variance_);
                end
            end
            
            if total_variance > 0
                obj.explained_variance_ratio_ = obj.explained_variance_ / total_variance;
            else
                obj.explained_variance_ratio_ = zeros(size(obj.explained_variance_));
            end
            
            % Store noise variance for reconstruction error
            total_components = min(obj.n_samples_, obj.n_features_);
            if num_components < total_components
                % Estimate noise variance from unexplained components
                if ~isempty(obj.singular_values_) && length(obj.singular_values_) == num_components
                    % For SVD, we need to compute total variance differently
                    total_var_estimate = trace(cov(zeros(obj.n_samples_, obj.n_features_))); % This is approximate
                    explained_var = sum(obj.explained_variance_);
                    obj.noise_variance_ = max(0, total_var_estimate - explained_var);
                else
                    % This is an approximation - in practice you might want to store all eigenvalues
                    obj.noise_variance_ = 0; % Will be computed during scoring if needed
                end
            else
                obj.noise_variance_ = 0;
            end
            
            % Overwrite n_components with the final integer value
            obj.n_components = num_components;
        end

        function X_transformed = transform(obj, X)
            % Apply dimensionality reduction to X.
            if isempty(obj.components_)
                error('PCA model has not been fitted yet. Call fit() first.');
            end
            
            % Validate input
            obj.validateInput(X, true); % Allow single samples for transform operations
            
            % Check feature dimension consistency
            if size(X, 2) ~= obj.n_features_
                error('X has %d features, but PCA was fitted with %d features.', ...
                      size(X, 2), obj.n_features_);
            end

            % Center the data using the mean from the fitted data
            X_centered = X - obj.mean_;
            
            % Project the data onto the principal components
            X_transformed = X_centered * obj.components_;
        end

        function X_original = inverse_transform(obj, X_transformed)
            % Transform data back to its original space.
            if isempty(obj.components_)
                error('PCA model has not been fitted yet. Call fit() first.');
            end
            
            % Validate input
            obj.validateInput(X_transformed, true); % Allow single samples for inverse transform
            
            % Check component dimension consistency
            if size(X_transformed, 2) ~= obj.n_components
                error('X_transformed has %d components, but PCA has %d components.', ...
                      size(X_transformed, 2), obj.n_components);
            end

            % Project back to the original feature space
            X_reconstructed_centered = X_transformed * obj.components_';
            
            % Add the mean back to get the final result
            X_original = X_reconstructed_centered + obj.mean_;
        end
        
        function X_transformed = fit_transform(obj, X)
            % Fit the model with X and apply the dimensionality reduction on X.
            obj = obj.fit(X);
            X_transformed = obj.transform(X);
        end
        
        function error_val = reconstruction_error(obj, X)
            % Compute the reconstruction error for the given data.
            % Returns the mean squared error between original and reconstructed data.
            
            if isempty(obj.components_)
                error('PCA model has not been fitted yet. Call fit() first.');
            end
            
            % Validate input
            obj.validateInput(X, true); % Allow single samples for reconstruction error calculation
            
            % Transform and inverse transform
            X_transformed = obj.transform(X);
            X_reconstructed = obj.inverse_transform(X_transformed);
            
            % Compute mean squared error
            diff = X - X_reconstructed;
            error_val = mean(diff(:).^2);
        end
        
        function scores = score_samples(obj, X)
            % Compute the log-likelihood of each sample under the model.
            % Useful for outlier detection in EEG data.
            
            if isempty(obj.components_)
                error('PCA model has not been fitted yet. Call fit() first.');
            end
            
            % Validate input
            obj.validateInput(X, true); % Allow single samples for scoring
            
            % Transform data
            X_transformed = obj.transform(X);
            
            % Compute precision matrix for PC space
            precision_pc = diag(1./obj.explained_variance_);
            
            % Compute scores (Mahalanobis distance in PC space)
            scores = zeros(size(X, 1), 1);
            for i = 1:size(X, 1)
                % Score based on PC components
                pc_score = X_transformed(i, :) * precision_pc * X_transformed(i, :)';
                
                % Add reconstruction error if we have fewer components than features
                if obj.n_components < obj.n_features_
                    % Compute reconstruction error for this single sample
                    X_single = X(i, :);
                    X_transformed_single = obj.transform(X_single);
                    X_reconstructed_single = obj.inverse_transform(X_transformed_single);
                    residual_error = sum((X_single - X_reconstructed_single).^2);
                    
                    % Add weighted residual error
                    if obj.noise_variance_ > 0
                        scores(i) = pc_score + residual_error / obj.noise_variance_;
                    else
                        scores(i) = pc_score + residual_error;
                    end
                else
                    scores(i) = pc_score;
                end
            end
        end
        
        function summary(obj)
            % Display a summary of the fitted PCA model.
            
            if isempty(obj.components_)
                fprintf('PCA model has not been fitted yet.\n');
                return;
            end
            
            fprintf('\n=== PCA Model Summary ===\n');
            fprintf('Decomposition method: %s\n', obj.svd_solver);
            fprintf('Number of samples: %d\n', obj.n_samples_);
            fprintf('Number of features: %d\n', obj.n_features_);
            fprintf('Number of components: %d\n', obj.n_components);
            fprintf('Total explained variance ratio: %.4f\n', sum(obj.explained_variance_ratio_));
            
            fprintf('\nExplained variance by component:\n');
            for i = 1:min(10, obj.n_components) % Show first 10 components
                fprintf('  PC%d: %.4f (%.2f%%)\n', i, obj.explained_variance_(i), ...
                        obj.explained_variance_ratio_(i) * 100);
            end
            
            if obj.n_components > 10
                fprintf('  ... (%d more components)\n', obj.n_components - 10);
            end
            
            if ~isempty(obj.singular_values_)
                fprintf('\nTop singular values:\n');
                for i = 1:min(5, length(obj.singular_values_))
                    fprintf('  ?%d: %.4f\n', i, obj.singular_values_(i));
                end
            end
            
            if obj.noise_variance_ > 0
                fprintf('Noise variance: %.4f\n', obj.noise_variance_);
            end
            fprintf('========================\n\n');
        end
        
        function [X_clean, outlier_scores] = detect_outliers(obj, X, threshold_percentile)
            % Detect and optionally remove outliers based on reconstruction error.
            % threshold_percentile: percentile threshold (e.g., 95 for top 5% outliers)
            
            if nargin < 3
                threshold_percentile = 95;
            end
            
            if isempty(obj.components_)
                error('PCA model has not been fitted yet. Call fit() first.');
            end
            
            % Validate input
            obj.validateInput(X);
            
            % Compute outlier scores
            outlier_scores = obj.score_samples(X);
            
            % Determine threshold
            threshold = prctile(outlier_scores, threshold_percentile);
            
            % Identify outliers
            is_outlier = outlier_scores > threshold;
            
            fprintf('Detected %d outliers (%.1f%% of data) using %dth percentile threshold.\n', ...
                    sum(is_outlier), 100*sum(is_outlier)/length(is_outlier), threshold_percentile);
            
            % Return clean data (without outliers)
            X_clean = X(~is_outlier, :);
        end
    end
    
    methods (Access = private)
        function validateInput(~, X, allow_single_sample)
            % Validate input data X.
            % allow_single_sample: if true, allows single sample (for internal use)
            
            if nargin < 3
                allow_single_sample = false;
            end
            
            % Check if X is numeric
            if ~isnumeric(X)
                error('Input data must be numeric.');
            end
            
            % Check for invalid values
            if any(isnan(X(:)))
                error('Input data contains NaN values. Please handle missing data before PCA.');
            end
            
            if any(isinf(X(:)))
                error('Input data contains infinite values. Please handle outliers before PCA.');
            end
            
            % Check if X is 2D
            if ~ismatrix(X) || ndims(X) ~= 2
                error('Input data must be a 2D matrix.');
            end
            
            % Check if X is empty
            if isempty(X)
                error('Input data is empty.');
            end
            
            % Check for minimum samples (only for fitting, not for transform operations)
            if ~allow_single_sample && size(X, 1) < 2
                error('PCA requires at least 2 samples.');
            end
            
            % Check for constant features (zero variance) - only for multi-sample data
            if size(X, 1) > 1
                feature_vars = var(X, 0, 1);
                if any(feature_vars == 0)
                    warning('One or more features have zero variance. Consider removing constant features.');
                end
            end
        end
    end
end