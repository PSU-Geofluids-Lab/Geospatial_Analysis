clear; clc;
close all;
% --- Target Parameters ---
tau_target = 0.3;        % tau = Xwf / (Xwf + Xf)
SE_target = 0.1;         % Standard error threshold on tau
Xf = 0.01;               % Fixed feedstock fraction
noise_sigma = 0.1;       % Log-normal noise (shape parameter)
rho_soil = 1.3;          % Soil bulk density (g/cm3)
rho_feed = 2.7;          % Feedstock (basalt) density (g/cm3)

Xwf_target = (tau_target * Xf) / (1 - tau_target);

M_i = 47.867; M_j = 40.078; % Molar masses (Ca and Ti)

soil_ppm.i = 10000; soil_ppm.j = 20000; % (made up, but reflective of data for Ca and Ti)
feed_ppm.i = 17000; feed_ppm.j = 80000;

soil.i = soil_ppm.i / (M_i * 1000); soil.j = soil_ppm.j / (M_j * 1000);
feedstock.i = feed_ppm.i / (M_i * 1000); feedstock.j = feed_ppm.j / (M_j * 1000);

max_iter = 100;
ticker = zeros(496, 2);
tau_tracker = zeros(496, 1);
tau_SE_tracker = zeros(496, 1);
ix = 1;

for n = 5:500
    xwf_list = zeros(max_iter,1);
    tau_list = zeros(max_iter,1);

    for k = 1:max_iter
        Xwf = Xwf_target; Xs = 1 - Xf - Xwf;

        j_mix = (Xs*rho_soil*soil.j + Xf*rho_feed*feedstock.j + Xwf*rho_soil*soil.j) / ...
                (Xs*rho_soil + Xf*rho_feed + Xwf*rho_soil);

        i_wf = soil.i + (rho_feed/rho_soil) * (feedstock.i - soil.i);
        i_mix = (Xs*rho_soil*soil.i + Xf*rho_feed*feedstock.i + Xwf*rho_soil*i_wf) / ...
                (Xs*rho_soil + Xf*rho_feed + Xwf*rho_soil);
        % This is where the averaging and mapping issues come into play
        samples.i = lognrnd(log(i_mix) - 0.5*noise_sigma^2, noise_sigma, [n, 1]);
        i_obs = mean(samples.i);

        obj = @(tau) (( (1 - Xf - (tau * Xf)/(1 - tau)) * rho_soil * soil.i + ...
                        Xf * rho_feed * feedstock.i + ...
                       ((tau * Xf)/(1 - tau)) * rho_soil * i_wf ) / ...
                      ( (1 - Xf - (tau * Xf)/(1 - tau)) * rho_soil + Xf * rho_feed + ((tau * Xf)/(1 - tau)) * rho_soil ) - i_obs )^2;

        tau_est = fminbnd(obj, 0.0001, 0.9999);
        tau_list(k) = tau_est;
        xwf_list(k) = (tau_est * Xf) / (1 - tau_est);
    end

    se_tau = std(tau_list);
    mean_xwf = mean(xwf_list);
    mean_tau = mean(tau_list);
    ticker(ix, :) = [n, mean_xwf];
    tau_tracker(ix) = mean_tau;
    tau_SE_tracker(ix) = se_tau;
    ix = ix + 1;

    if se_tau <= SE_target
        fprintf('Target SE of %.3f on tau reached at n = %d\n', SE_target, n);
        break;
    end
end

T = table(ticker(ix-1,1), mean_xwf, std(xwf_list), mean_tau, se_tau, ...
    'VariableNames', {'n_samples', 'mean_Xwf', 'SE_Xwf', 'mean_tau', 'SE_tau'});
disp('--- Summary of Final Estimates ---');
disp(T);

figure;
plot(ticker(1:ix-1,1), ticker(1:ix-1,2), '-o', 'LineWidth', 1.5);
xlabel('Sample size (n)'); ylabel('Mean X_{wf} estimate');
title('Progression of X_{wf} with increasing sample size'); grid on;

figure;
plot(ticker(1:ix-1,1), tau_tracker(1:ix-1), '-s', 'LineWidth', 1.5);
xlabel('Sample size (n)'); ylabel('Mean \tau estimate'); title('Progression of \tau with increasing sample size'); grid on;

figure;
subplot(1,2,1); histogram(xwf_list, 30); title('Distribution of X_{wf} estimates'); xlabel('X_{wf}'); ylabel('Frequency');
subplot(1,2,2); histogram(tau_list, 30); title('Distribution of \tau estimates'); xlabel('\tau'); ylabel('Frequency');

figure;
plot(ticker(1:ix-1,1), tau_SE_tracker(1:ix-1), 'LineWidth', 2);
xlabel('Sample size (n)'); ylabel('Standard Error on \tau');
title('Standard Error of \tau vs Sample Size'); grid on;

% Fit smoothed spline and compute derivative
n_vals = ticker(1:ix-1,1);
se_vals = tau_SE_tracker(1:ix-1);
pp = fit(n_vals, se_vals, 'smoothingspline');
dn = linspace(min(n_vals), max(n_vals), 1000);
dse = differentiate(pp, dn);

% Fit exponential model
f_exp = fittype('a*exp(-b*x) + c', 'independent', 'x');
fit_exp = fit(n_vals, se_vals, f_exp, 'StartPoint', [1, 0.01, 0.01]);
dse_exp = -fit_exp.a * fit_exp.b * exp(-fit_exp.b * dn);

% Find where |d(SE)/dn| crosses below 1.0
abs_dse = abs(dse_exp);
thresh = 1.0;
cross_idx = find(abs_dse(1:end-1) > thresh & abs_dse(2:end) < thresh, 1);

if ~isempty(cross_idx)
    x_thresh = dn(cross_idx + 1);
    y_thresh = dse_exp(cross_idx + 1);
else
    x_thresh = NaN;
    y_thresh = NaN;
end

figure;
subplot(2,1,1);
plot(n_vals, se_vals, 'o'); hold on;
plot(dn, pp(dn), 'r-', 'LineWidth', 2);
plot(dn, fit_exp(dn), 'g--', 'LineWidth', 1.5);
legend('Raw SE','Spline','Exponential Fit');
xlabel('Sample size (n)'); ylabel('SE on \tau'); title('Smoothed SE vs n'); grid on;

subplot(2,1,2);
plot(dn, dse_exp, 'b-', 'LineWidth', 2); hold on;
if ~isnan(x_thresh)
    plot(x_thresh, y_thresh, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot([x_thresh x_thresh], [0 y_thresh], 'k:');
    text(x_thresh + 2, y_thresh, sprintf('n = %d', ceil(x_thresh)), 'FontSize', 10, 'Color', 'k');
end
xlabel('Sample size (n)'); ylabel('d(SE)/dn'); title('First Derivative of Exponential Fit'); grid on;

abs_dse = abs(dse_exp);
thresh = 0.001;
cross_idx = find(abs_dse > thresh & [abs_dse(2:end), abs_dse(end)] <= thresh, 1);

if ~isempty(cross_idx)
    x_target = dn(cross_idx);
    y_target = dse_exp(cross_idx);
    se_at_target = fit_exp(x_target);
else
    x_target = NaN;
    y_target = NaN;
    se_at_target = NaN;
end

[se_min, min_idx] = min(pp(dn));
x_opt = dn(min_idx);
y_opt = se_min;

figure;
subplot(2,1,1);
plot(n_vals, se_vals, 'o'); hold on;
plot(dn, pp(dn), 'r-', 'LineWidth', 2);
plot(dn, fit_exp(dn), 'g--', 'LineWidth', 1.5);
if ~isnan(x_target)
    plot(x_target, se_at_target, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot([x_target x_target], [y_opt se_at_target], 'k:');
    text(x_target + 2, se_at_target, sprintf('target n = %d', ceil(x_target)), 'FontSize', 10, 'Color', 'k');
end
plot(x_opt, y_opt, 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
text(x_opt + 2, y_opt, sprintf('optimal n = %d', ceil(x_opt)), 'FontSize', 10, 'Color', 'b');

% Annotate SE difference
if ~isnan(x_target)
    delta_se = abs(se_at_target - y_opt);
    mid_x = (x_opt + x_target)/2;
    mid_y = (y_opt + se_at_target)/2;
    text(mid_x, mid_y, sprintf('ΔSE = %.4f', delta_se), ...
    'FontSize', 10, 'FontAngle', 'italic', 'HorizontalAlignment', 'center');
end

legend('Raw SE','Spline','Exponential Fit','Target Point','SE Drop','Optimal Point');
xlabel('Sample size (n)'); ylabel('SE on \tau'); title('Smoothed SE vs n'); grid on;

subplot(2,1,2);
plot(dn, dse_exp, 'b-', 'LineWidth', 2); hold on;
if ~isnan(x_target)
    plot(x_target, y_target, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot([x_target x_target], [0 y_target], 'k:');
    text(x_target + 2, y_target, sprintf('n = %d', ceil(x_target)), 'FontSize', 10, 'Color', 'k');
end
xlabel('Sample size (n)'); ylabel('d(SE)/dn'); title('First Derivative of Exponential Fit'); grid on;
