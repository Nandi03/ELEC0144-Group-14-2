IrisData;
[n, m] = size(IrisAttributesAndTypes);
i = randperm(n);
IrisDataJumbled = IrisAttributesAndTypes(i,:);

SeventyPercent = round(0.7*n, 0);
IrisDataTrain = IrisDataJumped(1:SeventyPercent, :);
IrisDataTest = IrisDataJumped(SeventyPercent+1:n, :);

x = IrisDataTrain(:, 1:4)';
y = IrisDataTrain(:, 5:7)';
