function mtrx_confusion = calc_confusion_matrix(vec_true_classes,vec_estimated_classes)
%CALC_CONFUSION_MATRI Calculates the confusion matrix from a set of
%classification results
mtrx_confusion = zeros(10);

for i=1:length(vec_estimated_classes)
    mtrx_confusion(vec_estimated_classes(i)+1, vec_true_classes(i)+1) = mtrx_confusion(vec_estimated_classes(i)+1, vec_true_classes(i)+1) + 1;
end
end

