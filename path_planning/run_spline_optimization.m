%% run_spline_optimization.m
% waypoints.csv -> spline optimization -> optimized_path_eps015.csv

clear; clc; close all;

%% ===== 설정 =====
ref_csv   = 'waypoints.csv';   % 입력 centerline
eps       = 0.00;                   % epsilon (0: min curvature, 1: shortest path)
margin    = 0.5;                    % 차폭 + 여유 (m), 경계에서 margin만큼 안쪽으로만 다님
out_csv   = sprintf('optimized_path_eps%03d.csv', round(eps*100));

%% ===== 데이터 로드 =====
data = readmatrix(ref_csv);

if size(data,2) >= 4
    % [x, y, z, width] 형식일 때
    xc     = data(:,1);
    yc     = data(:,2);
    width  = data(:,4);        % 도로 전체 폭 [m]
    half_w = width/2;
else
    % 폭 정보 없으면 8 m 가정
    xc     = data(:,1);
    yc     = data(:,2);
    half_w = 4 * ones(size(xc));  % 8 m / 2
end

% 닫힌 트랙이면 마지막 점 = 첫 점으로 맞춰줌
if norm([xc(end)-xc(1), yc(end)-yc(1)]) > 1e-3
    xc(end+1) = xc(1);
    yc(end+1) = yc(1);
    half_w(end+1) = half_w(1);
end

%% ===== 중심선 재샘플링 (옵션) =====
% 너무 듬성듬성하면 spline 계산이 불안정할 수 있어서
% 균일한 s 간격으로 다시 샘플링
s = [0; cumsum(hypot(diff(xc), diff(yc)))];
L = s(end);
ds = 0.1;                                % 1 m 간격
s_new = (0:ds:L).';

xc_s = interp1(s, xc, s_new, 'spline');
yc_s = interp1(s, yc, s_new, 'spline');
w_s  = interp1(s, half_w*2, s_new, 'linear');   % 전체 폭

%% ===== spline optimization 실행 =====
[xt_opt, yt_opt, alpha_opt] = optimizeSplineRacingPath(xc_s, yc_s, w_s, eps, margin);

%% ===== 결과 저장 =====
opt_path = [xt_opt, yt_opt];
writematrix(opt_path, out_csv);

fprintf('저장 완료: %s\n', out_csv);

%% ===== 결과 시각화 =====
figure; hold on; axis equal;
plot(xc_s, yc_s, 'k--', 'LineWidth', 1.0);           % centerline
plot(xt_opt, yt_opt, 'r-', 'LineWidth', 1.5);        % optimized path
legend('Reference centerline', 'Optimized racing line');
title(sprintf('Spline Optimization (\\epsilon = %.2f)', eps));
xlabel('x [m]'); ylabel('y [m]');
grid on;
