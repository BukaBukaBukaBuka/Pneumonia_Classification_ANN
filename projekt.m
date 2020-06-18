close all
clear all
clc

%% Sta³e dane
zdrowi = 1583;  chorzy = 4273; %doœæ wa¿ne
N = zdrowi + chorzy - 1; %ca³kowita liczba 
L = 10; %liczba niezale¿nych prób (wyci¹gana jest œrednia)
K = 5; %do walidacji krzy¿owej 5/10/(zdrowi+chorzy) - opcja leave-one-out

lr = 0.000001; %learning rate
P = 15; %liczba neuronów w warstwie ukrytej %!
show = 50; goal = 0; time = Inf; %parametry sieci 
epochs = 1000;

%% Wczytanie obrazów i ekstrakcja cech
% Z przetwarzaniem obrazów
addpath('images'); %z katalogu g³ównego
%{
for i = 1:zdrowi
    path = strcat('normal (', string(i), ').jpeg');
    image  = imread(path);
    if (size(image, 3) == 3)
        image = rgb2gray(image);
    end
    %image = dwt2(image) %Daubechies

    [glcm, SI] = graycomatrix(image, 'Offset', [0 1; -1 1; -1 0; -1 -1], 'NumLevels', 32, 'Symmetric', true);
    stats = graycoprops(glcm);
    staty = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
    Inputs(i, :) = staty;
    Targets(1, i) = 1;
end

for j = 1:chorzy
    path = strcat('pneumonia (', string(j), ').jpeg');
    image  = imread(path);
    if (size(image, 3) == 3)
        image = rgb2gray(image);
    end
    %image = dwt2(image) %Daubechies

    [glcm, SI] = graycomatrix(image, 'Offset', [0 1; -1 1; -1 0; -1 -1], 'NumLevels', 32, 'Symmetric', true);
    stats = graycoprops(glcm);
    staty = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
    Inputs(i+j, :) = staty;
    Targets(2, i+j) = 1;
end

Inputs = Inputs';
%}

% Zapisane dane
load('dane.mat');

%% Wybór danych
si = size(Targets); len = si(2); %si(1) - liczba klas - liczba wyjœæ
siz = size(Inputs); leng = siz(1); %licznoœæ zbioru wejœciowego

%walidacja krzy¿owa
for i = 1:N
    A(i) = mod(i, K) + 1;
end
A = A(randperm(N)); 

for k = 1:K
    Maska = zeros(1, N);
    for i = 1:N
        if A(i) == k
            Maska(i) = 1;
        end
    end
    Maska = logical(Maska);
    
    TargetMas = repmat(Maska, [si(1), 1]); 
    InputMas = repmat(Maska, [siz(1), 1]); 

    %podzia³ danych
    TargetUcz = Targets(TargetMas);             TargetUcz = reshape(TargetUcz, si(1), length(TargetUcz)/si(1));
    TargetTest = Targets(~TargetMas);           TargetTest = reshape(TargetTest, si(1), length(TargetTest)/si(1));
    InputUcz = Inputs(InputMas);                InputUcz = reshape(InputUcz, siz(1), length(InputUcz)/siz(1));
    InputTest = Inputs(~InputMas);              InputTest = reshape(InputTest, siz(1), length(InputTest)/siz(1));

    TU(k,:,:) = TargetUcz(:, :); TT(k,:,:) = TargetTest(:, :); 
    IU(k,:,:) = InputUcz(:, :);  IT(k,:,:) = InputTest(:, :);
end

%% Budowa sieci
ATT = []; ATU = []; AOT = []; AOU = [];
for p = 1:L %powtórzenia
    for k = 1:K %k=ty zbiór
        net = feedforwardnet(P); %liczba neuronów w warstwie ukrytej
        net = configure(net, squeeze(IU(k,:,:)), squeeze(TU(k,:,:))); %chyba zbêdne
        net.layers{1}.transferFcn = 'logsig'; %! - do zmiany logsig/hardlim/tansig
        net.layers{2}.transferFcn = 'purelin'; %domyœlnie
    
        net.IW{1} = rand(P, siz(1));
        net.LW{2,1} = rand(si(1), P);
        net.b{1} = rand(P, 1);
        net.b{2} = rand(si(1), 1);
        net.divideFcn = 'dividetrain';
        net.trainParam.show = show;
        net.trainParam.goal = goal;
        net.trainParam.time = time;
        net.trainParam.epochs = epochs;
    
        Tnet = train(net, squeeze(IU(k,:,:)), squeeze(TU(k,:,:)));
        OT(k,:,:) = sim(Tnet, squeeze(IT(k,:,:))); %output dla testowej
        OU(k,:,:) = sim(Tnet, squeeze(IU(k,:,:))); %output dla ucz¹cej

        perfT(p, k) = perform(Tnet, squeeze(TT(k,:,:)), squeeze(OT(k,:,:)));  
        perfU(p, k) = perform(Tnet, squeeze(TU(k,:,:)), squeeze(OU(k,:,:)));
        
        OT(k,:,:) = round(OT(k,:,:));
        OU(k,:,:) = round(OU(k,:,:));
        
        AOT = horzcat(AOT, squeeze(OT(k,:,:)));         ATT = horzcat(ATT, squeeze(TT(k,:,:))); 
        AOU = horzcat(AOU, squeeze(OU(k,:,:)));         ATU = horzcat(ATU, squeeze(TU(k,:,:)));
    end
end

perfTm = mean(mean(perfT));       perfTd = std(std(perfT)); 
perfUm = mean(mean(perfU));       perfUd = std(std(perfU));

figure(1); plotconfusion(ATU, AOU, 'Train');
figure(2); plotconfusion(ATT, AOT, 'Test');