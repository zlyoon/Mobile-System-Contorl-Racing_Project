function traj = generate_racing_path_from_waypoints(waypointFile, eps_val, start_pos, goal_pos)
% generate_racing_path_from_waypoints
%
%   waypointFile : centerline csv (x,y 헤더 포함 가정)
%   eps_val      : 0 ~ 1 (0 = 최소 곡률, 1 = 최단 거리)
%   start_pos    : [x,y] 시작 좌표
%   goal_pos     : [x,y] 도착 좌표
%
%   출력:
%     traj       : [N x 2] (x,y), start_pos ~ goal_pos 구간 racing line

    %% ---------------- 0. 입력 기본값 ----------------
    if nargin < 1 || isempty(waypointFile)
        waypointFile = "waypoints-2.csv";
    end
    if nargin < 2 || isempty(eps_val)
        eps_val = 0.15;
    end
    if nargin < 3 || isempty(start_pos)
        start_pos = [-154.6, 18.0];
    end
    if nargin < 4 || isempty(goal_pos)
        goal_pos  = [-159.7, 23.0];
    end

    %% ---------------- 1. centerline 로드 ----------------
    T  = readtable(waypointFile);
    xc = T.x(:);
    yc = T.y(:);

    % 폐곡선 보장 (마지막 점과 처음 점이 다르면 한 점 더 추가)
    if norm([xc(1)-xc(end), yc(1)-yc(end)]) > 1e-3
        xc(end+1) = xc(1);
        yc(end+1) = yc(1);
    end

    n = numel(xc);

    %% ---------------- 2. 도로 폭 / 유효 폭 ----------------
    road_width    = 7.0;   % [m] 공지 도로 폭
    vehicle_width = 1.2;   % [m]
    margin        = 2.0;   % [m] 한쪽 여유 (실제로는 차폭/2 + margin 만큼 띄운다고 보면 됨)

    % centerline 기준 한쪽으로 쓸 수 있는 half-width
    usable_half = road_width/2 - (vehicle_width/2 + margin);
    if usable_half <= 0
        error('margin/vehicle_width가 너무 커서 유효 폭이 0 이하입니다.');
    end

    % 최적화 함수에 넘길 유효 half-width (양쪽 동일)
    tw = usable_half * ones(n,1);

    % 전체 도로 폭 half (시각화용)
    full_half = (road_width/2) * ones(n,1);

    %% ---------------- 2-1. 법선 방향 / 전체 경계 계산 ----------------
    % 약간 smoothing해서 법선 방향 튐 방지
    dx = gradient(xc);
    dy = gradient(yc);
    if n > 5
        k = ones(5,1)/5;
        dx = conv(dx, k, 'same');
        dy = conv(dy, k, 'same');
    end
    dL = hypot(dx,dy) + 1e-6;

    xoff = @(a) -a .* dy ./ dL + xc;
    yoff = @(a)  a .* dx ./ dL + yc;

    % 전체 도로 폭 경계
    xin_full  = xoff(-full_half);
    yin_full  = yoff(-full_half);
    xout_full = xoff( full_half);
    yout_full = yoff( full_half);

    %% ---------------- 3. Racing line 최적화 ----------------
    % 이 함수는 "폐곡선 전체"에 대해 최적 racing line을 계산한다고 가정
    % 시그니처 예:
    %   [traj_full, xin, yin, xout, yout, i_start, i_goal] = ...
    %       optimizeSplineRacingPathfinal(xc, yc, tw, eps_val, start_pos, goal_pos);
    %
    [traj_full, xin_all, yin_all, xout_all, yout_all] = ...
        optimizeSplineRacingPathfinal(xc, yc, tw, eps_val, start_pos, goal_pos);

    % traj_full : [M x 2] 폐곡선 racing line (한 바퀴 전체)

    %% ---------------- 4. start_pos ~ goal_pos 구간만 잘라내기 ----------------
    % traj_full 안에서 start/goal에 가장 가까운 인덱스 찾기
    dist_s = vecnorm(traj_full - start_pos, 2, 2);
    dist_g = vecnorm(traj_full - goal_pos,  2, 2);
    [~, idx_s] = min(dist_s);
    [~, idx_g] = min(dist_g);

    % 폐곡선이므로 idx_s → idx_g 방향으로 한 바퀴를 따라가도록 잘라냄
    if idx_g < idx_s
        % goal이 index 상으로 앞에 있으면 끝까지 갔다가 처음으로 wrap
        traj = [traj_full(idx_s:end, :); traj_full(1:idx_g, :)];
        xin  = [xin_all(idx_s:end);      xin_all(1:idx_g)];
        yin  = [yin_all(idx_s:end);      yin_all(1:idx_g)];
        xout = [xout_all(idx_s:end);     xout_all(1:idx_g)];
        yout = [yout_all(idx_s:end);     yout_all(1:idx_g)];
    else
        traj = traj_full(idx_s:idx_g, :);
        xin  = xin_all(idx_s:idx_g);
        yin  = yin_all(idx_s:idx_g);
        xout = xout_all(idx_s:idx_g);
        yout = yout_all(idx_s:idx_g);
    end

    % 맨 앞/뒤 점만 정확히 start_pos, goal_pos로 덮어쓰기
    traj(1,:)   = start_pos;
    traj(end,:) = goal_pos;

    %% ---------------- 5. CSV 저장 ----------------
    outName = sprintf("path_eps%03d_margin%03d.csv", ...
                      round(eps_val*100), round(margin*100));
    writematrix(traj, outName);
    fprintf(">> Saved optimized racing line (start→goal) to %s\n", outName);

    %% ---------------- 6. 시각화 ----------------
    figure; hold on; grid on; axis equal;
    title(sprintf("Racing Map Visualization (eps=%.3f)", eps_val));
    xlabel("x [m]"); ylabel("y [m]");

    % (1) centerline
    plot(xc, yc, 'k--', 'LineWidth', 1.0);

    % (2) full road width boundary (회색)
    plot(xin_full,  yin_full,  'Color',[0.6 0.6 0.6], 'LineWidth',1.0);
    plot(xout_full, yout_full, 'Color',[0.6 0.6 0.6], 'LineWidth',1.0);

    % (3) effective boundary (blue/red 점선)
    plot(xin,  yin,  'b:', 'LineWidth',1.4);
    plot(xout, yout, 'r:', 'LineWidth',1.4);

    % (4) 최종 racing line (start→goal 구간)
    plot(traj(:,1), traj(:,2), 'g', 'LineWidth',2.5);

    % (5) start / goal 표시
    plot(start_pos(1), start_pos(2), 'mo', 'MarkerSize', 8, 'LineWidth',2);
    text(start_pos(1), start_pos(2), " start", 'Color','m', 'HorizontalAlignment','left');

    plot(goal_pos(1), goal_pos(2), 'co', 'MarkerSize', 8, 'LineWidth',2);
    text(goal_pos(1), goal_pos(2), " goal", 'Color','c', 'HorizontalAlignment','left');

    legend("centerline", ...
           "full inner", "full outer", ...
           "effective inner", "effective outer", ...
           "optimized racing line", ...
           "start", "goal", ...
           'Location','bestoutside');
end
