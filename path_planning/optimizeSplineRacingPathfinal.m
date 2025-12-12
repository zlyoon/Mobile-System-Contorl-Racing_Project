function [traj, xin, yin, xout, yout, i_start, i_goal] = optimizeSplineRacingPath( ...
    xc, yc, tw, eps_val, start_pos, goal_pos)

    xc = xc(:); yc = yc(:);
    twr = tw(:); twl = tw(:);

    n = numel(xc);

    %% -------------------------------------------------------
    % 1) Compute boundaries
    %% -------------------------------------------------------
    dx = gradient(xc);
    dy = gradient(yc);
    dL = hypot(dx, dy) + 1e-6;

    xoff = @(a) -a .* dy ./ dL + xc;
    yoff = @(a)  a .* dx ./ dL + yc;

    xin = xoff(-twr);
    yin = yoff(-twr);
    xout = xoff(twl);
    yout = yoff(twl);

    delx = xout - xin;
    dely = yout - yin;

    %% -------------------------------------------------------
    % 2) QP cost matrices (same as before)
    %% -------------------------------------------------------
    H1 = zeros(n); B1=zeros(n,1);
    H2 = zeros(n); B2=zeros(n,1);

    for i = 2:n-1
        H1(i-1,i-1) = H1(i-1,i-1) + delx(i-1)^2 + dely(i-1)^2;
        H1(i-1,i)   = H1(i-1,i)   - 2*(delx(i-1)*delx(i) + dely(i-1)*dely(i));
        H1(i-1,i+1) = H1(i-1,i+1) + delx(i-1)*delx(i+1) + dely(i-1)*dely(i+1);
        H1(i,i-1) = H1(i-1,i);
        H1(i,i)   = H1(i,i) + 4*(delx(i)^2 + dely(i)^2);
        H1(i,i+1) = H1(i,i+1) - 2*(delx(i)*delx(i+1) + dely(i)*dely(i+1));
        H1(i+1,i-1) = H1(i-1,i+1);
        H1(i+1,i)   = H1(i,i+1);
        H1(i+1,i+1) = H1(i+1,i+1) + delx(i+1)^2 + dely(i+1)^2;
    end

    for i = 2:n-1
        ddx = xin(i+1)+xin(i-1)-2*xin(i);
        ddy = yin(i+1)+yin(i-1)-2*yin(i);
        B1(i-1)=B1(i-1)+2*(ddx*delx(i-1)+ddy*dely(i-1));
        B1(i)  =B1(i)  -4*(ddx*delx(i)  +ddy*dely(i));
        B1(i+1)=B1(i+1)+2*(ddx*delx(i+1)+ddy*dely(i+1));
    end

    for i = 1:n-1
        H2(i,i)     = H2(i,i) + delx(i)^2 + dely(i)^2;
        H2(i,i+1)   = H2(i,i+1) - (delx(i)*delx(i+1) + dely(i)*dely(i+1));
        H2(i+1,i)   = H2(i,i+1);
        H2(i+1,i+1) = H2(i+1,i+1) + delx(i+1)^2 + dely(i+1)^2;
    end

    for i = 1:n-1
        ddx = xin(i+1)-xin(i);
        ddy = yin(i+1)-yin(i);
        B2(i)   = B2(i)   - 2*(ddx*delx(i) + ddy*dely(i));
        B2(i+1) = B2(i+1) + 2*(ddx*delx(i+1) + ddy*dely(i+1));
    end

    %% -------------------------------------------------------
    % 3) Final cost
    %% -------------------------------------------------------
    H = (1-eps_val)*H1 + eps_val*H2;
    B = (1-eps_val)*B1 + eps_val*B2;

    lb = zeros(n,1);
    ub = ones(n,1);

    Aeq = zeros(1,n); Aeq(1)=1; Aeq(end)=-1; beq=0;

    %% -------------------------------------------------------
    % 4) Determine start/goal indices
    %% -------------------------------------------------------
    [~, i_start] = min((xc-start_pos(1)).^2 + (yc-start_pos(2)).^2);
    [~, i_goal]  = min((xc-goal_pos(1)).^2  + (yc-goal_pos(2)).^2);

    % 둘 다 centerline 근처 α=0.4~0.6 제약
    win = 3;
    idx_s = max(1, i_start-win):min(n, i_start+win);
    idx_g = max(1, i_goal-win):min(n, i_goal+win);

    lb(idx_s)=0.4; ub(idx_s)=0.6;
    lb(idx_g)=0.4; ub(idx_g)=0.6;

    %% -------------------------------------------------------
    % 5) Solve QP
    %% -------------------------------------------------------
    opts = optimoptions("quadprog","Display","off");
    alpha = quadprog(2*H, B, [], [], Aeq, beq, lb, ub, [], opts);

    if isempty(alpha)
        alpha = 0.5 * ones(n,1);
    end

    %% -------------------------------------------------------
    % 6) Build final trajectory
    %% -------------------------------------------------------
    x_opt = xin + alpha .* delx;
    y_opt = yin + alpha .* dely;
    traj = [x_opt, y_opt];

end
