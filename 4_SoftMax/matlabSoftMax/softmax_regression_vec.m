function [ Jw, dJw] = softmax_regression_vec( w, X, y )
    % Alumn: Javier Vargas
    
    % Unwrapping data
    [M,N] = size(X);
    Q = length(w)/N + 1;
    w = reshape(w, N, Q-1);
    
    % Calculating cost function
        XW = X*w;                   % matrix of x(n) w(q)

        eXW = exp(XW);              % exponential of the matrix
        eXW = [eXW ones(M,1)];

        I = sub2ind(size(eXW), [1:M]', y);
        Jw = -1/M * sum( log(eXW(I) ./ sum(eXW,2)) );

        % This is another ineficcient way to solve it
        %     tic;
        %     Jw = 0;
        %     for m = 1:M;
        %         for q = 1:Q;
        %            % We use the minus "-" cause of the Jw = - sum(...)
        %             Jw = Jw - (Y(m) == q) * log ( eXW(m,q) / sum(eXW(m,:)) );
        %         end
        %     end
        %     toc;

    % Calculating the gradient
        Aux = bsxfun(@eq, y, [1:Q]);
        eXWN = bsxfun(@rdivide, eXW, sum(eXW,2));
        g = -1/M * X' * (Aux - eXWN);
        g = g(:);
        dJw = g(1:(Q-1)*N);

end

