function gradSoft = softmaxGradient(x)
    sigma = softmax(x);
    gradSoft = zeros(size(x));
    for i = 1:length(x)
        for j = 1:length(x)
            if i == j
                gradSoft(i,j) = sigma(i)*(1-sigma(i));
            else
                gradSoft(i,j) = -sigma(i)*sigma(j);
            end
        end
    end
end