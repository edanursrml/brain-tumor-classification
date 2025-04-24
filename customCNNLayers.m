function layers = customCNNLayers(classNames, classWeights)

inputSize = [224 224 1];
in = imageInputLayer(inputSize, 'Normalization', 'zerocenter', 'Name', 'input');

% === Yardımcı fonksiyonlar ===
conv = @(n,k,s,name) convolution2dLayer(k,n,'Padding','same','Stride',s,'Name',name);
bn   = @(name) batchNormalizationLayer('Name',name);
relu = @(name) reluLayer('Name',name);
mp   = @(name) maxPooling2dLayer(2,'Stride',2,'Name',name);

layers = [
    in

    conv(64,3,1,'conv1')
    bn('bn1')
    relu('relu1')
    mp('pool1')

conv(128,3,1,'conv3')
bn('bn3')
relu('relu3')
mp('pool3')

conv(256,3,1,'conv4')
bn('bn4')
relu('relu4')
mp('pool4')

dropoutLayer(0.2,'Name','drop')
fullyConnectedLayer(128,'Name','fc1')
batchNormalizationLayer('Name','bn_fc1')
relu('relu_fc1')
fullyConnectedLayer(numel(classNames),'Name','fc2')
softmaxLayer('Name','softmax')
classificationLayer('Name','output', ...
    'Classes', classNames, ...
    'ClassWeights', classWeights)
];

end
