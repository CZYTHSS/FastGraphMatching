function runLAPJV(datafile)
    fin = fopen(datafile, 'r');
    K = fscanf(fin, '%d\n', [1]);
    format = '%f';
    for i = 1:K-1
        format = strcat([format, ',%f']);
    end
    A = fscanf(fin, format, [K Inf]);
    [rowsol, cost] = lapjv(A);
    
    rowsol
    cost
end

