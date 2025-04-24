for i = 30:60
    rng(i);  % Sabit seed
    % Eğitim işlemleri
    layers = myAdvancedCNN(classNames, classWeights);
    net = trainNetwork(augTrain, layers, options);
    save(sprintf('model_seed%d.mat', i), 'net');
    % Eğitim sonrası
    YPred = classify(net, augTest);
    YTest = imdsTest.Labels;
    
    % Skor hesapla
    acc = sum(YPred == YTest) / numel(YTest);
    confMat = confusionmat(YTest, YPred);
    TP = confMat(2,2); FP = confMat(1,2); FN = confMat(2,1);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);
    
    % Ekrana yaz
    fprintf("Test Accuracy: %.2f%%\n", acc * 100);
    fprintf("Precision: %.2f%%\n", precision * 100);
    fprintf("Recall: %.2f%%\n", recall * 100);
    fprintf("F1-score: %.2f%%\n", f1 * 100);
% Tahmin edilen tüm örnekleri tek tek yazdır
    fprintf("\n--- Tahmin Detayları (seed = %d) ---\n", i);
    for j = 1:numel(YTest)
        trueLabel = string(YTest(j));
        predictedLabel = string(YPred(j));
        if trueLabel == predictedLabel
            fprintf("✅ %03d — Doğru | Gerçek: %-10s | Tahmin: %-10s\n", j, trueLabel, predictedLabel);
        else
            fprintf("❌ %03d — Yanlış | Gerçek: %-10s | Tahmin: %-10s\n", j, trueLabel, predictedLabel);
        end
    end

end
