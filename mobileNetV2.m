% Veri kümesinin yüklenmesi
dataDir = 'brain_tumor_dataset';
allImages = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Veriyi eğitim, doğrulama ve test olarak ayır
[trainData, restData] = splitEachLabel(allImages, 0.7, 'randomized');
[valData, testData] = splitEachLabel(restData, 0.5, 'randomized');

% Veri artırma (augmentation) ayarları
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXReflection', true);

% Önceden eğitilmiş MobileNetV2 modelini yükle
pretrainedNet = mobilenetv2;

% Ağ katmanlarını düzenlemek için layerGraph formatına dönüştür
layerGraphNet = layerGraph(pretrainedNet);

% Önceden eğitilmiş modelin orijinal çıkış katmanlarını kaldır
layerGraphNet = removeLayers(layerGraphNet, {'Logits','Logits_softmax','ClassificationLayer_Logits'});

% Yeni sınıflandırma katmanları bloğu oluştur (tam bağlantılı, aktivasyon vb.)
newLayersBlock = [
    fullyConnectedLayer(128, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.5, 'Name', 'drop1')

    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')

    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

% Yeni katmanları ana ağa ekle ve bağla
layerGraphNet = addLayers(layerGraphNet, newLayersBlock);
layerGraphNet = connectLayers(layerGraphNet, 'global_average_pooling2d_1', 'fc1');

% İnce ayar (fine-tuning) için son 30 katmanın öğrenme oranlarını artır
numLayers = numel(layerGraphNet.Layers);
for layerIdx = numLayers-29:numLayers
    layerToTune = layerGraphNet.Layers(layerIdx);
    if isprop(layerToTune, 'WeightLearnRateFactor')
        layerToTune.WeightLearnRateFactor = 2;
        layerToTune.BiasLearnRateFactor   = 2;
        layerGraphNet = replaceLayer(layerGraphNet, layerToTune.Name, layerToTune);
    end
end

% Ağ giriş boyutunu al
inputSize = pretrainedNet.Layers(1).InputSize;
inputImageSize = inputSize(1:2);

% Eğitim, doğrulama ve test verilerini ağ giriş boyutuna uygun ve renkli formata getir
augTrainDS = augmentedImageDatastore(inputImageSize, trainData, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augValDS   = augmentedImageDatastore(inputImageSize, valData, ...
    'ColorPreprocessing', 'gray2rgb');
augTestDS  = augmentedImageDatastore(inputImageSize, testData, ...
    'ColorPreprocessing', 'gray2rgb');

% Eğitim seçeneklerini belirle
trainOpts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augValDS, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Ağı eğit (transfer öğrenimi ile ince ayar)
fineTunedNet = trainNetwork(augTrainDS, layerGraphNet, trainOpts);

% Test verisiyle modeli değerlendir ve sonuçları görselleştir
predictedLabels = classify(fineTunedNet, augTestDS);
trueLabels      = testData.Labels;
confusionchart(trueLabels, predictedLabels);

% Eğitilmiş modeli kaydet
save('mobileNetV2_Trained.mat', 'fineTunedNet', 'inputSize');
