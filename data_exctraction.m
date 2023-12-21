%% --- Program initialization ---
% 
% This program creates a class with a specified amount of astrocytoma
% microscopy images for use in the classification algorithms

clc;
ImageNumber = input('Enter the number of images for training: ');
TumorType = input('Enter the grade type of the training set [0](LOW)/[1](HIGH): ');
f = zeros(ImageNumber, 27);

for i=1:ImageNumber % Run for the number of images requested
% Reading input image
file = uigetfile('*.*');
dataIn = imread(file);
figure; imshow(dataIn);
title('Original Image')

%% --- Illumination Correction ---

% Illumination correction function
% Image must be RGB
[dataOut,errSurface,errSurfaceMin,errSurfaceMax] = shadingCorrection(dataIn, 500);

% Converting data to the right format
im = dataOut/255;
% j = imagesc(im);
% colorbar

im = im2gray(im);

figure; imshow(im)
title('Illumination Corrected Image')

%% --- Nuclei Segmentation ---

% Increasing of image contrast
im2 = imadjust(im);
% imshow(im2)

% Reversing the colors of the image (BW)
J = imcomplement(im2);
% imshow(J)

% Gauss filtering to smooth the image
gauss = imgaussfilt(J, 3);
% figure; imshow(gauss)

%Edge detection in image
Edges = edge(gauss, 'Canny');
% imshow(Edges)

% Filling the area of the nuclei edges
filled = imfill(Edges,'holes');
% imshow(filled)


% Morphological Opening to clear out the rest of the edges
open = imopen(filled, strel('disk', 5));
% imshow(open)

% Opening to clear out the small objects
I = bwareaopen(open, 300);
% imshow(I)

% Clearing of the nuclei that overlap in the borders
I = imclearborder(I);
% figure; imshow(I)


% % Removal of connected nuclei
% % Filtering out nuclei with Eccentricity > 0.89 and Solidity < 0.95
bw_out = bwpropfilt(I, 'Eccentricity', [-Inf, 0.89]);

bw_out = bwpropfilt(bw_out, 'Solidity', [0.95, Inf]);
% figure; imshow(bw_out)

% Smoothing of segmented nuclei
windowSize=3;  % Decide as per your requirements
kernel=ones(windowSize)/windowSize^2;
result=conv2(single(bw_out),kernel,'same');
result=result>0.5;
bw_out(~result)=0; 
figure, imshow(bw_out);
title('Mask (Segmented Nuclei)')

% Edge detection to show results of segmentation
edge2 = edge(bw_out, 'Canny');
% imshow(edge2)

C = imfuse(edge2, im2, 'blend');
imshow(C)
title('Segmentation Results')

% Overlaying mask with original image to show only the nuclei
Inew = im.*bw_out;
imshow(Inew)
title('Final Masked Image with Segmented Nuclei')
Inew = im2uint8(Inew);

%% --- Feature Extraction ---

% Nuclei labeling for morphological features
% L is the map that shows the coordinates of each object
% num is the number of total objects in image
[L, num] = bwlabel(bw_out,8);

% Calculation of 8 Morphological Features
% Area, Circularity, EquivDiameter, Perimeter, Eccentricity, Extent,
% MajorAxisLength, MinorAxisLength
area = regionprops(bw_out, 'Area'); 
circ = regionprops(bw_out, 'Circularity');  
diam = regionprops(bw_out, 'EquivDiameter');
per = regionprops(bw_out, 'Perimeter');
ecc = regionprops(bw_out, 'Eccentricity');
conva = regionprops(bw_out, 'ConvexArea');
majorAL = regionprops(bw_out, 'MajorAxisLength');
minorAL = regionprops(bw_out, 'MinorAxisLength');

% Calculation of the mean value of each Morphological Feature
m_area = mean([area.Area]);
m_circ = mean([circ.Circularity]);
m_diam = mean([diam.EquivDiameter]);
m_per = mean([per.Perimeter]);
m_ecc = mean([ecc.Eccentricity]);
m_conva = mean([conva.ConvexArea]);
m_majorAL = mean([majorAL.MajorAxisLength]);
m_minorAL = mean([minorAL.MinorAxisLength]);

% Calculation of image histogram
H = imhist(Inew);
% Removing value of the first element which is the black pixels 
H(1) = 0;


% Calculation of Texture Features

% 1st order Texture Features (Mean Value, SD, Skewness, Kurtosis)
mv = mean(H);
sd = std(H);
sk = skewness(H);
ku = kurtosis(H);

fprintf("---Morphological Features Mean Values---\nArea: %f\nCircularity: %f\nEquivDiameter: %f\nPerimeter: %f\nEccentricity: %f\nConvexArea: %f\nMajorAxisLength: %f\nMinorAxisLength: %f\n", m_area, m_circ, m_diam, m_per, m_ecc, m_conva, m_majorAL, m_minorAL);


% Calculation of the 4 Co-occurrence matrices (0, 90, 45, 135)
NumLevels = 32;
GrayLimits = [0, 255];
P0  = graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[0,1]) + graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[0,-1]); % GLCM for 0 degrees
P90 = graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[1,0]) + graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[-1,0]); % GLCM for 90 degrees
P45 = graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[1,-1])+ graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[-1,1]); % GLCM for 45 degrees
P135= graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[1,1])+ graycomatrix(Inew,'GrayLimits', GrayLimits,'NumLevels',NumLevels,'Offset',[-1,-1]); % GLCM for 135 degrees

% Calculation of 4 features (Contrast, Correlation, Energy, Homogeneity) for every matrix 
f0 = graycoprops(P0); f90 = graycoprops(P90); f45 = graycoprops(P45); f135 = graycoprops(P135);

% Mean value of each feature
Con_m = (f0.Contrast + f90.Contrast + f45.Contrast +f135.Contrast)/4;
Cor_m = (f0.Correlation + f90.Correlation + f45.Correlation + f135.Correlation)/4;
Enr_m = (f0.Energy + f90.Energy + f45.Energy + f135.Energy)/4;
Hom_m = (f0.Homogeneity + f90.Homogeneity + f45.Homogeneity + f135.Homogeneity)/4;

% Range of each feature 
Con_r = max([f0.Contrast, f90.Contrast, f45.Contrast, f135.Contrast]) - min([f0.Contrast, f90.Contrast, f45.Contrast, f135.Contrast]);
Cor_r = max([f0.Correlation, f90.Correlation, f45.Correlation, f135.Correlation]) - min([f0.Correlation, f90.Correlation, f45.Correlation, f135.Correlation]);
Enr_r = max([f0.Energy, f90.Energy, f45.Energy, f135.Energy]) - min([f0.Energy, f90.Energy, f45.Energy, f135.Energy]);
Hom_r = max([f0.Homogeneity, f90.Homogeneity, f45.Homogeneity, f135.Homogeneity]) - min([f0.Homogeneity, f90.Homogeneity, f45.Homogeneity, f135.Homogeneity]);


fprintf("---Texture Features---\n")
fprintf("---1st Order---\nMean: %f\nSD: %f\nSkewness: %f\nKurtosis: %f", mv, sd, sk, ku)
fprintf("---2nd Order---\n---co-occurrence matrix---")
fprintf("--Mean values--\nContrast: %f\nCorrelation: %f\nEnergy: %f\nHomogeneity: %f\n", Con_m, Cor_m, Enr_m, Hom_m)
fprintf("--Range values--\nContrast: %f\nCorrelation: %f\nEnergy: %f\nHomogeneity: %f\n", Con_r, Cor_r, Enr_r, Hom_r)



% Calculation of run-length matrix
mask = ones(size(Inew(:,:,1))); % mask equal to the size of image

[SRE,LRE,GLN,RP,RLN,LGRE,HGRE]  = glrlm(Inew,16,mask);

fprintf("--- Run-Length Matrix---")
fprintf("SRE: %f\nLRE: %f\nGLN: %f\nRP: %f\nRLN: %f\nLGRE: %f\nHGRE: %f\n", SRE, LRE,GLN, RP, RLN, LGRE, HGRE)

% Saving features to an array
f(i,1) = m_area;
f(i,2) = m_circ;
f(i,3) = m_diam;
f(i,4) = m_per;
f(i,5) = m_ecc;
f(i,6) = m_conva;
f(i,7) = m_majorAL;
f(i,8) = m_minorAL;
f(i,9) = mv;
f(i,10) = sd;
f(i,11) = sk;
f(i,12) = ku;
f(i,13) = Con_m;
f(i,14) = Con_r;
f(i,15) = Cor_m;
f(i,16) = Cor_r;
f(i,17) = Enr_m;
f(i,18) = Enr_r;
f(i,19) = Hom_m;
f(i,20) = Hom_r;
f(i,21) = SRE;
f(i,22) = LRE;
f(i,23) = GLN;
f(i,24) = RP;
f(i,25) = RLN;
f(i,26) = LGRE;
f(i,27) = HGRE;
end


if TumorType == 0
    save('FeaturesLow.txt', 'f', '-ascii')
elseif TumorType == 1
    save('FeaturesHigh.txt', 'f', '-ascii')
else
    fprintf("Invalid input! Re-run the program and select either 0 (LOW) or 1 (HIGH):")
end
