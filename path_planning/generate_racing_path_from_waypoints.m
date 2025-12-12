function traj = generate_racing_path_from_waypoints(waypointFile, eps_val, start_pos)
% generate_racing_path_from_waypoints
%   waypointFile : waypoints.csv (x,y 헤더 포함)
%   eps_val      : 0~1 사이 epsilon (0: 최소 곡률, 1: 최단 거리)
%   start_pos    : [x0, y0], 차량 리스폰 좌표 (옵션)
%
%   traj         : [x_opt, y_opt] (최적 레이싱 라인)

    if nargin < 1
        waypointFile = "waypoints.csv";
    end
    if nargin < 2
        eps_val = 0.15;   % 기본값 (곡률 위주)
    end
    if nargin < 3
        % 기본 리스폰 좌표 (필요하면 여기 바꾸기)
        start_pos = [-20, -24];
    end

    %% -------------------------------------------------------
    %  1. waypoints.csv 로드 (중앙선)
    % --------------------------------------------------------
    T = readtable(waypointFile);  % 1행이 'x','y' 헤더
    x_center = T.x;
    y_center = T.y;

    x_center = x_center(:);
    y_center = y_center(:);

    % 폐곡선이 아니면 닫아줌
    if norm([x_center(1)-x_center(end), y_center(1)-y_center(end)]) > 1e-3
        x_center(end+1) = x_center(1);
        y_center(end+1) = y_center(1);
    end

    %% -------------------------------------------------------
    %  2. 차량 규격 + 도로 폭 가정 → 유효 폭 계산
    % --------------------------------------------------------
    % 도로 전체 폭 가정 (필요하면 여기 숫자만 바꾸면 됨)
    lane_width = 8.0;                   % [m] 전체 차선 폭(가정)
    road_width = lane_width * ones(size(x_center));

    vehicle_width = 1.2;                % [m] 차량 폭
    safety_margin = 1.7;                % [m] 각쪽 여유

    % 실제 주행 가능한 유효 폭
    effective_width = road_width - vehicle_width - 2*safety_margin;
    effective_width = max(effective_width, 2.5);  % 너무 좁아지지 않게 최소폭 보장

    % 중앙선 기준 좌/우 half width
    twr = effective_width/2;   % right side half width
    twl = effective_width/2;   % left  side half width

    %% -------------------------------------------------------
    %  3. Spline Optimization으로 최적 경로 계산
    % --------------------------------------------------------
    [traj, xin, yin, xout, yout, alpha] = optimizeSplineRacingPath( ...
        x_center, y_center, twr, twl, eps_val, start_pos);

    % 결과 저장 (ROS2에서 waypoint로 사용)
    outName = sprintf('optimized_path_eps%03d_mg1.7.csv', round(eps_val*100));
    writematrix(traj, outName);

    fprintf('>> Saved optimized path to "%s"\n', outName);

    %% -------------------------------------------------------
    %  4. 확인용 플롯
    % --------------------------------------------------------
    figure; hold on; axis equal; grid on;
    plot(x_center, y_center, 'k--', 'LineWidth', 1.0);  % 중앙선
    plot(xin, yin, 'b:', 'LineWidth', 1.0);             % 안쪽 경계
    plot(xout, yout, 'r:', 'LineWidth', 1.0);           % 바깥 경계
    plot(traj(:,1), traj(:,2), 'g', 'LineWidth', 2.0);  % 최적 경로
    legend('center line','inner boundary','outer boundary', ...
           sprintf('optimized (\\epsilon=%.2f)', eps_val), ...
           'Location','best');
    xlabel('x [m]'); ylabel('y [m]');
    title('Spline Optimization Racing Line (from waypoints)');

    % 시작점 표시 (옵션)
    plot(start_pos(1), start_pos(2), 'mo', 'MarkerSize', 8, 'LineWidth', 2);
    text(start_pos(1), start_pos(2), '  start pos', 'Color', 'm');

end


function [traj, xin, yin, xout, yout, alpha] = optimizeSplineRacingPath( ...
    xc, yc, twr, twl, eps_val, start_pos)
% xc, yc : centerline
% twr, twl : center에서 오른쪽/왼쪽 경계까지 half width
% eps_val : (0~1) 0=minimum curvature, 1=shortest path
% start_pos : [x0, y0], 시작 위치 근처는 centerline에 가깝게 제약
%
% traj : [x_opt, y_opt]

    % 열벡터로 정리
    xc  = xc(:);
    yc  = yc(:);
    twr = twr(:);
    twl = twl(:);

    n = numel(xc);

    %% 1) 중앙선의 tangent, 길이
    dx = gradient(xc);
    dy = gradient(yc);
    dL = hypot(dx, dy) + 1e-6;   % 0 나누기 방지

    % 법선 방향으로 offset
    xoff = @(a) -a .* dy ./ dL + xc;
    yoff = @(a)  a .* dx ./ dL + yc;

    % 안쪽/바깥쪽 경계
    xin  = xoff(-twr);
    yin  = yoff(-twr);
    xout = xoff(twl);
    yout = yoff(twl);

    % 안쪽→바깥쪽 벡터
    delx = xout - xin;
    dely = yout - yin;

    %% 2) 최소 곡률 / 최단 거리 비용 행렬 H1, H2 구성
    H1 = zeros(n);   % curvature 관련
    B1 = zeros(n,1);

    H2 = zeros(n);   % path length 관련
    B2 = zeros(n,1);

    % ---- (1) H1, B1 : 최소 곡률
    for i = 2:n-1
        H1(i-1,i-1) = H1(i-1,i-1) + delx(i-1)^2 + dely(i-1)^2;
        H1(i-1,i)   = H1(i-1,i)   - 2*delx(i-1)*delx(i) - 2*dely(i-1)*dely(i);
        H1(i-1,i+1) = H1(i-1,i+1) + delx(i-1)*delx(i+1) + dely(i-1)*dely(i+1);

        H1(i,i-1)   = H1(i,i-1)   - 2*delx(i-1)*delx(i) - 2*dely(i-1)*dely(i);
        H1(i,i)     = H1(i,i)     + 4*delx(i)^2 + 4*dely(i)^2;
        H1(i,i+1)   = H1(i,i+1)   - 2*delx(i)*delx(i+1) - 2*dely(i)*dely(i+1);

        H1(i+1,i-1) = H1(i+1,i-1) + delx(i-1)*delx(i+1) + dely(i-1)*dely(i+1);
        H1(i+1,i)   = H1(i+1,i)   - 2*delx(i)*delx(i+1) - 2*dely(i)*dely(i+1);
        H1(i+1,i+1) = H1(i+1,i+1) + delx(i+1)^2 + dely(i+1)^2;
    end

    for i = 2:n-1
        ddx = xin(i+1) + xin(i-1) - 2*xin(i);
        ddy = yin(i+1) + yin(i-1) - 2*yin(i);

        B1(i-1) = B1(i-1) + 2*ddx*delx(i-1) + 2*ddy*dely(i-1);
        B1(i)   = B1(i)   - 4*ddx*delx(i)   - 4*ddy*dely(i);
        B1(i+1) = B1(i+1) + 2*ddx*delx(i+1) + 2*ddy*dely(i+1);
    end

    % ---- (2) H2, B2 : 최단 거리
    for i = 1:n-1
        H2(i,i)     = H2(i,i)     + delx(i)^2 + dely(i)^2;
        H2(i+1,i)   = H2(i+1,i)   - delx(i)*delx(i+1) - dely(i)*dely(i+1);
        H2(i,i+1)   = H2(i,i+1)   - delx(i)*delx(i+1) - dely(i)*dely(i+1);
        H2(i+1,i+1) = H2(i+1,i+1) + delx(i+1)^2 + dely(i+1)^2;
    end

    for i = 1:n-1
        ddx = xin(i+1) - xin(i);
        ddy = yin(i+1) - yin(i);

        B2(i)   = B2(i)   - 2*ddx*delx(i)   - 2*ddy*dely(i);
        B2(i+1) = B2(i+1) + 2*ddx*delx(i+1) + 2*ddy*dely(i+1);
    end

    %% 3) eps로 두 비용을 섞어서 최종 QP 구성
    H = (1-eps_val)*H1 + eps_val*H2;
    B = (1-eps_val)*B1 + eps_val*B2;

    % alpha ∈ [0,1] : 0이면 안쪽 경계, 1이면 바깥 경계
    %amin = 0.2;
    %amax = 0.9;
    %lb = amin * ones(n,1);
    %ub = amax * ones(n,1);
    
    lb = zeros(n,1);
    ub = ones(n,1);

    % 폐곡선: alpha(1) = alpha(n) 유지
    Aeq = zeros(1,n);
    Aeq(1)   = 1;
    Aeq(end) = -1;
    beq = 0;

    % === 시작 위치 근처는 centerline에 더 가깝게 제약 ===
    if ~isempty(start_pos)
        dx0 = xc - start_pos(1);
        dy0 = yc - start_pos(2);
        [~, i0] = min(dx0.^2 + dy0.^2);   % 시작 위치에 가장 가까운 index

        idx_win = (i0-2):(i0+2);          % 주변 5개 정도 묶어서 제약
        idx_win = idx_win(idx_win >= 1 & idx_win <= n);

        % 중앙선 근처 (0.4~0.6)로 alpha 제한
        lb(idx_win) = max(lb(idx_win), 0.4);
        ub(idx_win) = min(ub(idx_win), 0.6);

        fprintf('>> Constraining alpha near start index %d, window [%d..%d]\n', ...
            i0, idx_win(1), idx_win(end));
    end

    opts = optimoptions('quadprog','Display','off');

    % quadprog는 0.5*x'Hx + f'x 최소 → 여기서는 x'Hx + B'x라서 H에 2곱
    alpha = quadprog(2*H, B, [], [], Aeq, beq, lb, ub, [], opts);

    if isempty(alpha)
        warning('quadprog failed, using alpha = 0.5 (center line)');
        alpha = 0.5 * ones(n,1);
    end

    %% 4) 최적 경로 계산
    x_opt = xin + alpha .* delx;
    y_opt = yin + alpha .* dely;

    traj = [x_opt, y_opt];
end
