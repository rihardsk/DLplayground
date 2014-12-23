function [opttheta, cost] = pso(cost, bounds, options)
    maxiter = 100
    k = 0.5; % in [0, 1]
    fi = 15; % > 4
    w = 2 * k / abs(2 - fi - sqrt(fi^2 - 4 * fi));
    vmax = 4;
    side = 6;
    pcount = side^2;
    r = rand(pcount, size(bounds, 1));
    x = (bounds(:, 2) - bounds(:, 1))' .* r + bounds(:, 1)';
    v = zeros(size(x));
    xcost = inf(size(x, 1), 1);
    pbest = x;
    pbestcost = xcost;
    gbest = pbest;
    gbestcost = pbestcost;
    iter = 0;
    while iter < maxiter
        cells = num2cell(x, 2);
        xcost = reshape(cellfun(cost, cells), side, side);
        
        mask = xcost(:) < pbestcost;
        pbest(mask, :) = x(mask, :);
        pbestcost(mask) = xcost(mask);
        gbestind = get_gbest(pbestcost, side, @min);
        gbest = pbest(gbestind, :);
        
        fip = rand() * 2;
        fig = rand() * 2;
        %fig = 1 - fip;
        fi = 1;
        w = 1;
        v = w * (v + fi * (fip * (pbest - x) + fig * (gbest - x)));
        v = max(v, -vmax);
        v = min(v, vmax);
        vabs = sqrt(sum(v.^2));
        x = x + v;
        
        iter = iter + 1;
        fprintf("Iter %i, cost = %f, vmin=%f, vmax=%f, vmean=%f, better=%i\n", iter, min(pbestcost(:)), min(vabs), max(vabs), mean(vabs), numel(mask(mask)));
        fflush(stdout);
    end
    [cost, ind] = min(pbestcost(:));
    opttheta = pbest(ind, :)';
end

function r = iroot(n)
    r = 0;
    q = n + 1;
    while r + 1 < q
        p = floor((q + r)/2);
        if n < p^2
            q=p;
        else
            r=p;
        end
    end
end
