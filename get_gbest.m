function inds = get_gbest(xcost, diff, fun)
    sides = size(xcost);
    inds = zeros(sides);
    for j = 1:sides(2)
        for i = 1:sides(1)
            lbound = max(j - diff, 1);
            rbound = min(j + diff, sides(2));
            tbound = max(i - diff, 1);
            bbound = min(i + diff, sides(1));
            neighbors = xcost(tbound:bbound, lbound:rbound);
            [m, ind] = fun(neighbors(:));
            indi = mod(ind - 1, bbound - tbound + 1);
            indj = floor((ind - 1) / (bbound - tbound + 1));
            inds(i, j) = (tbound + indi) + (lbound + indj - 1) * sides(1);
        end
     end
end