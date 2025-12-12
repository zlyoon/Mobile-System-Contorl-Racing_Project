function [xtraj, ytraj, alpha_opt] = optimizeSplineRacingPath(xc, yc, width, eps, margin)
% optimizeSplineRacingPath
%   xc, yc : centerline (uniform spacing 권장)
%   width  : 도로 전체 폭 [m] (각 점마다 값 가능)
%   eps    : 0 ~ 1, 0: min curvature, 1: shortest path
%   margin : 차폭 + 여유 [m] (경계에서 margin 만큼 안쪽만 사용)
%
%   출력:
%     xtraj, ytraj : 최적 경로 (same size as xc/yc)
%     alpha_opt    : [0,1] 사이의 lateral interpolation 계수
%
%   방식:
%     - centerline 기준 tangent, normal 계산
%     - inner / outer boundary 생성
%     - QP:  (1-eps)*J_curvature + eps*J_length 최소화
%            변수: a(i) \in [0,1],  P_i = P_in_i + a_i * (P_out_i - P_in_i)

    n = numel(xc);
    if numel(yc) ~= n || numel(width) ~= n
        error('xc, yc, width 길이가 같아야 합니다.');
    end

    % --- 1. 경계 생성 (inner / outer) ---
    % 중심선 tangent
    dx = gradient(xc);
    dy = gradient(yc);
    dL = hypot(dx, dy) + 1e-9;  % 0으로 나누기 방지

    % 법선 벡터 (왼쪽 방향)
    nx = -dy ./ dL;
    ny =  dx ./ dL;

    % 사용 가능한 반폭 (road_half - margin)
    half_w = width / 2;
    usable = max(half_w - margin, 0.1);  % 최소 0.1 m (너무 0에 가까우면 수치 불안정)

    % inner / outer (임의로 "왼쪽 = in", "오른쪽 = out" 으로 정의)
    xin = xc - usable .* nx;
    yin = yc - usable .* ny;
    xout = xc + usable .* nx;
    yout = yc + usable .* ny;

    % delx, dely: inner -> outer 방향 벡터
    delx = xout - xin;
    dely = yout - yin;

    % --- 2. QP 행렬 초기화 ---
    % 변수: a(1)...a(n)  (각각 [0,1])
    H1 = zeros(n);   % min curvature term
    B1 = zeros(1,n);
    H2 = zeros(n);   % shortest path term
    B2 = zeros(1,n);

    %% ===== (1) Min curvature term (H1, B1) =====
    %   discrete 2차 미분(두 번 차분)을 이용한 곡률 근사 기반
    %   보고서의 loop 구조를 참고하여 구성 :contentReference[oaicite:2]{index=2}
    for i = 2:n-1
        % 편의를 위해 인덱스를 한 줄에서 보기 좋게
        im1 = i-1; ip1 = i+1;

        % --- Hessian (H1) ---
        % first row
        H1(im1,im1) = H1(im1,im1) + delx(im1)^2         + dely(im1)^2;
        H1(im1,i)   = H1(im1,i)   - 2*delx(im1)*delx(i) - 2*dely(im1)*dely(i);
        H1(im1,ip1) = H1(im1,ip1) + delx(im1)*delx(ip1) + dely(im1)*dely(ip1);

        % second row
        H1(i,im1)   = H1(i,im1)   - 2*delx(im1)*delx(i) - 2*dely(im1)*dely(i);
        H1(i,i)     = H1(i,i)     + 4*delx(i)^2         + 4*dely(i)^2;
        H1(i,ip1)   = H1(i,ip1)   - 2*delx(i)*delx(ip1) - 2*dely(i)*dely(ip1);

        % third row
        H1(ip1,im1) = H1(ip1,im1) + delx(im1)*delx(ip1) + dely(im1)*dely(ip1);
        H1(ip1,i)   = H1(ip1,i)   - 2*delx(i)*delx(ip1) - 2*dely(i)*dely(ip1);
        H1(ip1,ip1) = H1(ip1,ip1) + delx(ip1)^2         + dely(ip1)^2;

        % --- Gradient (B1) ---
        % (xin, yin)을 기준으로 두 번 차분 값 이용
        ddx = xin(ip1) + xin(im1) - 2*xin(i);
        ddy = yin(ip1) + yin(im1) - 2*yin(i);

        B1(1,im1) = B1(1,im1) + 2*ddx*delx(im1) + 2*ddy*dely(im1);
        B1(1,i)   = B1(1,i)   - 4*ddx*delx(i)   - 4*ddy*dely(i);
        B1(1,ip1) = B1(1,ip1) + 2*ddx*delx(ip1) + 2*ddy*dely(ip1);
    end

    %% ===== (2) Shortest path term (H2, B2) =====
    %   segment 길이의 제곱을 최소화하는 형태 :contentReference[oaicite:3]{index=3}
    for i = 1:n-1
        ip1 = i+1;

        % Hessian
        H2(i,i)     = H2(i,i)     + delx(i)^2        + dely(i)^2;
        H2(ip1,i)   = H2(ip1,i)   - delx(i)*delx(ip1) - dely(i)*dely(ip1);
        H2(i,ip1)   = H2(i,ip1)   - delx(i)*delx(ip1) - dely(i)*dely(ip1);
        H2(ip1,ip1) = H2(ip1,ip1) + delx(ip1)^2      + dely(ip1)^2;

        % Gradient
        ddx = xout(ip1) - xout(i);    % 여기서는 inner/outer 중 어떤 기준을 쓸지 다양한 변형 가능
        ddy = yout(ip1) - yout(i);

        B2(1,i)   = B2(1,i)   - 2*ddx*delx(i)   - 2*ddy*dely(i);
        B2(1,ip1) = B2(1,ip1) + 2*ddx*delx(ip1) + 2*ddy*dely(ip1);
    end

    %% ===== (3) 최종 QP 행렬 =====
    H = (1-eps)*H1 + eps*H2;
    B = (1-eps)*B1 + eps*B2;

    % 대칭화 (수치 오차 방지)
    H = 0.5*(H + H.');

    %% ===== (4) 경계 조건 및 제약 =====
    % 변수 a(i)는 [0,1] (inner~outer)
    %amin = 0.2;
    %amax = 0.9;
    %lb = amin * ones(n,1);
    %ub = amax * ones(n,1);


    lb = zeros(n,1);
    ub = ones(n,1);

    % 시작점과 끝점을 centerline에 가깝게 고정 (a=0.5)
    Aeq = zeros(2,n);
    beq = zeros(2,1);

    Aeq(1,1)   = 1;   beq(1)   = 0.5;
    Aeq(2,end) = 1;   beq(2)   = 0.5;

    % quadprog:  1/2 x'Hx + f'x  →  여기서는 f = B'
    options = optimoptions('quadprog', ...
                           'Display','none', ...
                           'Algorithm','interior-point-convex');

    alpha_opt = quadprog(2*H, B.', [], [], Aeq, beq, lb, ub, [], options);

    %% ===== (5) 최종 경로 계산 =====
    xtraj = xin + alpha_opt .* delx;
    ytraj = yin + alpha_opt .* dely;
end
