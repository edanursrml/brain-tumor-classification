clc; clear; close all;
% Veri setini içe aktar ve görüntü deposu oluştur
% rng(423);  % Tekrarlanabilirlik için sabit tohum (isteğe bağlı)
rng('shuffle');  % Farklı sonuçlar için rastgele tohum kullan
s = rng;

dataDir = 'brain_tumor_dataset';
allImages = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Tüm görüntüleri gri tonlamaya çevir ve 224x224 boyutuna getir
allImages.ReadFcn = @(x) imresize(im2single(im2gray(imread(x))), [224, 224]);

% Veriyi eğitim ve kalan kısım olarak ayır
[trainData, restData] = splitEachLabel(allImages, 0.7, 'randomized');

% 'No' sınıfını oversampling ile iki katına çıkar (azınlık sınıfı artırma)
isNo = trainData.Labels == "no";
isYes = trainData.Labels == "yes";

noTrainData = subset(trainData, find(isNo));
yesTrainData = subset(trainData, find(isYes));

dupNoFiles  = [noTrainData.Files;  noTrainData.Files(1:floor(end/2))];
dupNoLabels = [noTrainData.Labels; noTrainData.Labels(1:floor(end/2))];

% Oversampling sonrasında yeni eğitim verisini oluştur
combinedTrainFiles  = [dupNoFiles; yesTrainData.Files];
combinedTrainLabels = [dupNoLabels; yesTrainData.Labels];

trainData = imageDatastore(combinedTrainFiles, 'Labels', combinedTrainLabels);

% Eğitim veri deposu için ReadFcn'i tekrar tanımla (ön işleme garantisi)
trainData.ReadFcn = @(x) imresize(im2single(im2gray(imread(x))), [224, 224]);

% Kalan veriyi doğrulama ve test olarak ayır
[valData, testData] = splitEachLabel(restData, 0.5, 'randomized');

% Veri artırma (augmentation) işlemlerini tanımla
augmenter = imageDataAugmenter( ...
    'RandRotation',    [-20, 20], ...
    'RandXTranslation',[-5, 5], ...
    'RandYTranslation',[-5, 5], ...
    'RandXScale',      [0.9, 1.1], ...
    'RandYScale',      [0.9, 1.1], ...
    'RandXShear',      [-10, 10], ...
    'RandYShear',      [-10, 10], ...
    'RandXReflection', true, ...
    'RandYReflection', true);

augTrainDS = augmentedImageDatastore([224 224 1], trainData, 'DataAugmentation', augmenter);
augValDS   = augmentedImageDatastore([224 224 1], valData);
augTestDS  = augmentedImageDatastore([224 224 1], testData);

% Ağ katmanlarını oluştur
classes = categories(trainData.Labels);
classWeightsVec = [1.5, 2];  % (İsteğe bağlı) Veri dengesine göre ayarlanabilir
cnnLayers = customCNNLayers(classes, classWeightsVec);

% Eğitim ayarlarını (hiperparametreleri) belirle
trainOpts = trainingOptions("adam", ...
    "InitialLearnRate",    3e-4, ...
    "LearnRateSchedule",   "piecewise", ...
    "LearnRateDropPeriod", 10, ...
    "LearnRateDropFactor", 0.2, ...
    "ValidationPatience",  5, ...
    "MaxEpochs",           50, ...
    "MiniBatchSize",       64, ...
    "Shuffle",            "every-epoch", ...
    "ValidationData",      augValDS, ...
    "ValidationFrequency", 5, ...
    "ExecutionEnvironment","auto", ...
    "Verbose",            true, ...
    "Plots",             "training-progress");

% Ağı eğit
trainedNetwork = trainNetwork(augTrainDS, cnnLayers, trainOpts);

% Modeli kaydet
save('customCNN_Trained.mat', 'trainedNetwork');

% Test verisi ile tahmin yap ve doğruluk hesapla
predictedLabels = classify(trainedNetwork, augTestDS);
trueLabels     = testData.Labels;
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Karışıklık matrisi ve performans metrikleri
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix - Custom CNN');

confMatrix    = confusionmat(trueLabels, predictedLabels);
truePositive  = confMatrix(2, 2); 
falsePositive = confMatrix(1, 2); 
falseNegative = confMatrix(2, 1);
precision = truePositive / (truePositive + falsePositive + eps);
recall    = truePositive / (truePositive + falseNegative + eps);
f1Score   = 2 * (precision * recall) / (precision + recall + eps);

fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n',    recall * 100);
fprintf('F1-score: %.2f%%\n',  f1Score * 100);

fprintf('Kullanılan rastgele tohum: %d\n', s.Seed);

analyzeNetwork(trainedNetwork);
