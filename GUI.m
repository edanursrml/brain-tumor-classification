function GUI()
    uiFig = uifigure('Name', 'CNN Tahmin Arayüzü', 'Position', [100 100 500 350]);

    modelDropdown = uidropdown(uiFig, ...
        'Items', {'Ensemble', 'Model 1', 'Model 2', 'Model 3', 'MobileNet'}, ...
        'Position', [30 280 150 30]);

    imageAxes = uiaxes(uiFig, 'Position', [220 120 250 200]);
    title(imageAxes, 'Yüklenen Görüntü');
    imshow(ones(224, 224), 'Parent', imageAxes);

    resultLabel = uilabel(uiFig, 'Position', [30 200 400 30], 'FontSize', 14, 'Text', 'Tahmin:');

    uibutton(uiFig, 'Text', 'Görsel Yükle & Tahmin Et', ...
        'Position', [30 240 150 30], ...
        'ButtonPushedFcn', @(btn, event) onPredict(modelDropdown, imageAxes));

    imagePath = '';
    modelDropdown.ValueChangedFcn = @(dd, event) autoPredictImage(imageAxes);

    persistent net1 net2 net3 netMobile inputSize;
    if isempty(net1)
        net1 = load('customCNN_Trained.mat').net;
        net2 = load('customCNN_Trained2.mat').net;
        net3 = load('customCNN_seed_320072433.mat').net;
        m = load('mobileNetV2_Trained.mat');
        netMobile = m.netTLFinetuned;
        inputSize = m.inputSize;
    end

    function onPredict(modelDropdown, imageAxes)
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Bir görsel seçin');
        if isequal(file, 0); return; end
        imagePath = fullfile(path, file);
        img = imread(imagePath);
        imshow(img, 'Parent', imageAxes);
        title(imageAxes, 'Yüklenmiş Görüntü');
        classifyImage(imagePath, modelDropdown.Value);
    end

    function autoPredictImage(imageAxes)
        if ~isempty(imagePath) && isfile(imagePath)
            img = imread(imagePath);
            imshow(img, 'Parent', imageAxes);
            title(imageAxes, 'Yüklenmiş Görüntü');
            classifyImage(imagePath, modelDropdown.Value);
        end
    end

    function classifyImage(imgPath, method)
        img = imread(imgPath);
        switch method
            case 'Model 1'
                inputImage = preprocessGray(img);
                probs = predict(net1, inputImage);
                classes = net1.Layers(end).Classes;
                showResult(probs, classes, 'Model 1');

            case 'Model 2'
                inputImage = preprocessGray(img);
                probs = predict(net2, inputImage);
                classes = net2.Layers(end).Classes;
                showResult(probs, classes, 'Model 2');

            case 'Model 3'
                inputImage = preprocessGray(img);
                probs = predict(net3, inputImage);
                classes = net3.Layers(end).Classes;
                showResult(probs, classes, 'Model 3');

            case 'MobileNet'
                if size(img, 3) == 1
                    img = cat(3, img, img, img);
                end
                resizedImg = imresize(img, inputSize(1:2));
                probs = predict(netMobile, resizedImg);
                classes = netMobile.Layers(end).Classes;
                showResult(probs, classes, 'MobileNet');

            case 'Ensemble'
                inputImage = preprocessGray(img);
                p1 = predict(net1, inputImage);
                p2 = predict(net2, inputImage);
                p3 = predict(net3, inputImage);
                avgProbs = (p1 + p2 + p3) / 3;
                classes = net1.Layers(end).Classes;
                showResult(avgProbs, classes, 'Ensemble');
        end
    end

    function inputImage = preprocessGray(img)
        grayImage = im2gray(img);
        resizedImage = imresize(grayImage, [224 224]);
        inputImage = im2single(resizedImage);
        inputImage = reshape(inputImage, [224 224 1]);
    end

    function showResult(probabilities, classList, modelName)
        [maxProb, idx] = max(probabilities);
        predictedLabel = classList(idx);
        resultLabel.Text = sprintf('⚡ Tahmin: %s (%s, %.1f%%)', ...
            string(predictedLabel), modelName, maxProb * 100);
    end
end
