function dist_eucl = calc_distance_euclidian(x,mu)
%CALC_DISTANCE_EUCLIDIAN Calculates the euclidian distance between a
%feature vector x and a reference vector mu

% x, mu must have equal size
assert(size(x,1) == size(mu,1),'dist_eucl: feature length =! reference length');

dist_eucl = (x-mu)'*(x-mu);
end

