function img = brightContrast(imagePath)
   % Görüntüyü okuyup gri tonlamaya dönüştür
    outImg = im2gray(imread(imagePath));

    outImg = im2single(outImg);
    outImg = imresize(outImg, [224 224]);

    % Görüntülerin sadece %20'sine adaptif histogram eşitleme uygula
    if rand < 0.2
        outImg = adapthisteq(outImg, 'ClipLimit', 0.01);
    else
        % Kalanlarına kontrast germe işlemi uygula
        limits = stretchlim(outImg, [0.05 0.95]);
        outImg = imadjust(outImg, limits, []);
    end

    % Çıktıyı [224 x 224 x 1] boyutunda bir matris haline getir
    outImg = reshape(outImg, [224 224 1]);
end
