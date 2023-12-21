%% --- Program initialization ---
% 
% This program reads a histopathological image obtained from a microscope,
% corrects the shading, creates a nuclei segmentation in order to obtain texture and
% morphological features and then classifies the image in 2 categories of 
% low grade or high grade 

% Reading input image
% clear all;
warning off all

% Select image file to classify
file = uigetfile('*.*');

if file == 0
% if user cancels the input, end the program
    return;
end

dataIn = imread(file);
figure; imshow(dataIn);
title('Original Image')

%% --- Shading Correction ---

% Illumination correction function
% Image must be RGB
[dataOut,errSurface,errSurfaceMin,errSurfaceMax] = shadingCorrection(dataIn, 500);

% Converting data to the right format
im = dataOut/255;
% j = imagesc(im);
% colorbar
figure;
imshow(im)

% Convert image to grayscale
im = im2gray(im);
figure; imshow(im)
title('Illumination Corrected Grayscale Image')

%% --- Nuclei Segmentation ---

% Increasing of image contrast
im2 = imadjust(im);
imshow(im2)

% Reversing the colors of the image (BW)
J = imcomplement(im2);
imshow(J)

% Gauss filtering to smooth the image
gauss = imgaussfilt(J, 3);
figure; imshow(gauss)

%Edge detection in image
Edges = edge(gauss, 'Canny');
imshow(Edges)

% Filling the area of the nuclei edges
filled = imfill(Edges,'holes');
imshow(filled)


% Morphological Opening to clear out unnecessary edges
open = imopen(filled, strel('disk', 5));
imshow(open)

% Opening to clear out the small objects
I = bwareaopen(open, 300);
imshow(I)

% Clearing of the nuclei that overlap in the borders
I = imclearborder(I);
figure; imshow(I)


% % Removal of connected nuclei
% % Filtering out nuclei with Eccentricity > 0.89 and Solidity < 0.95
bw_out = bwpropfilt(I, 'Eccentricity', [-Inf, 0.89]);

bw_out = bwpropfilt(bw_out, 'Solidity', [0.95, Inf]);
figure; imshow(bw_out)

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
imshow(edge2)

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
f(1,1) = m_area;
f(1,2) = m_circ;
f(1,3) = m_diam;
f(1,4) = m_per;
f(1,5) = m_ecc;
f(1,6) = m_conva;
f(1,7) = m_majorAL;
f(1,8) = m_minorAL;
f(1,9) = mv;
f(1,10) = sd;
f(1,11) = sk;
f(1,12) = ku;
f(1,13) = Con_m;
f(1,14) = Con_r;
f(1,15) = Cor_m;
f(1,16) = Cor_r;
f(1,17) = Enr_m;
f(1,18) = Enr_r;
f(1,19) = Hom_m;
f(1,20) = Hom_r;
f(1,21) = SRE;
f(1,22) = LRE;
f(1,23) = GLN;
f(1,24) = RP;
f(1,25) = RLN;
f(1,26) = LGRE;
f(1,27) = HGRE;

%% Dataset 
% classes and data preparation

class1= [ % microscopy images of brain tissue (Astrocytoma) HighGrade 

   8.7196000e+02   9.7206220e-01   3.2491752e+01   1.0416296e+02   6.5012896e-01   8.9408000e+02   3.9385099e+01   2.7481302e+01   8.5152344e+01   7.8799001e+01   5.4881368e-01   1.7523320e+00   3.4501486e-01   1.4551639e-01   9.1302159e-01   3.6623406e-02   9.7970316e-01   4.2213570e-04   9.9581012e-01   1.1419851e-03   4.4922650e-01   2.1932760e+05   5.7114954e+03   2.0937822e-02   9.0202377e+03   5.1229132e+01   5.7114954e+03
   1.1142174e+03   9.6044675e-01   3.6214471e+01   1.1657530e+02   6.4967628e-01   1.1464348e+03   4.3216142e+01   3.1134358e+01   1.0010547e+02   8.8970719e+01   2.7773957e-01   1.3853707e+00   3.8776142e-01   1.4754799e-01   9.3036190e-01   2.6446813e-02   9.7629261e-01   4.1522487e-04   9.9487156e-01   1.1482648e-03   4.7953115e-01   1.7549578e+05   6.2678209e+03   2.5547357e-02   1.2600356e+04   6.7751175e+01   6.2678209e+03
   9.6342857e+02   9.6874746e-01   3.4558668e+01   1.1065800e+02   6.2662661e-01   9.9171429e+02   4.0825398e+01   3.0051387e+01   2.6343750e+01   5.5986824e+01   2.2186092e+00   6.7524277e+00   5.1351277e-01   1.9272584e-01   9.4559395e-01   2.0350625e-02   9.6833074e-01   5.8613544e-04   9.9314766e-01   1.5101511e-03   4.8190054e-01   4.9329666e+04   3.4846186e+03   4.2331272e-02   4.1951841e+03   1.2992855e+02   3.4846186e+03
   8.9233333e+02   1.0094583e+00   3.2459920e+01   1.0182111e+02   5.9537390e-01   9.1244444e+02   3.6909495e+01   2.8957933e+01   3.1371094e+01   6.2235534e+01   2.0159563e+00   5.7324397e+00   4.9775624e-01   1.7631030e-01   9.4156647e-01   2.0623736e-02   9.6235066e-01   6.8136328e-04   9.9241160e-01   1.7436896e-03   4.6925629e-01   4.4337079e+04   3.4656051e+03   4.6343768e-02   4.3805392e+03   1.0873255e+02   3.4656051e+03
   1.0446000e+03   9.6220751e-01   3.4537165e+01   1.1255160e+02   6.0465628e-01   1.0750000e+03   4.2213466e+01   2.9356033e+01   2.0402344e+01   4.6967304e+01   2.6684604e+00   9.5393202e+00   3.7967872e-01   1.4908482e-01   9.4936443e-01   1.9814683e-02   9.7549381e-01   4.4425751e-04   9.9489083e-01   1.1842181e-03   4.3535099e-01   6.8127422e+04   2.9664919e+03   3.3117224e-02   2.7560669e+03   1.2040007e+02   2.9664919e+03
   9.4636000e+02   9.6590845e-01   3.3606432e+01   1.0795760e+02   6.4824934e-01   9.7368000e+02   3.9946876e+01   2.9024519e+01   9.2417969e+01   9.9409714e+01   6.9935935e-01   1.9455949e+00   3.3268363e-01   1.2094826e-01   9.1292777e-01   3.1606181e-02   9.7802616e-01   3.8713422e-04   9.9485374e-01   1.1517119e-03   5.0305700e-01   1.7853950e+05   6.8499523e+03   2.5855463e-02   1.3923195e+04   5.2561905e+01   6.8499523e+03
   9.2258333e+02   1.0002564e+00   3.2917594e+01   1.0377883e+02   6.0189656e-01   9.4208333e+02   3.7971580e+01   2.9073251e+01   4.3246094e+01   3.5975236e+01   5.5741093e-01   2.0004098e+00   1.8073520e-01   7.7892018e-02   9.2694600e-01   3.1442548e-02   9.8969219e-01   2.0169302e-04   9.9743091e-01   6.0562143e-04   4.7686116e-01   3.4706877e+05   4.2858030e+03   1.4647544e-02   7.1207784e+03   5.6666463e+01   4.2858030e+03
   8.8933333e+02   9.6960741e-01   3.2558593e+01   1.0448717e+02   6.5181782e-01   9.1566667e+02   3.9043995e+01   2.7788234e+01   4.1687500e+01   4.6127317e+01   8.2730421e-01   2.0836166e+00   1.6513709e-01   6.2378715e-02   9.2153378e-01   2.9582483e-02   9.9004410e-01   1.8211818e-04   9.9789773e-01   5.3492182e-04   3.7484369e-01   4.1837585e+05   4.5757563e+03   1.2143401e-02   3.8988148e+03   4.8884243e+01   4.5757563e+03
   6.6939535e+02   9.8117088e-01   2.8869404e+01   9.1950930e+01   6.6238736e-01   6.8674419e+02   3.4813014e+01   2.4516200e+01   1.1243750e+02   1.5862256e+02   1.1295589e+00   2.7369651e+00   2.1800920e+00   9.9364795e-01   9.1594323e-01   3.8206932e-02   8.6730368e-01   3.4138555e-03   9.6584871e-01   6.5122797e-03   5.9303780e-01   7.4768931e+03   9.8910141e+03   1.7010046e-01   2.4881095e+04   8.3080003e+01   9.8910141e+03
   7.5857143e+02   1.0090121e+00   3.0581620e+01   9.5792857e+01   6.5201120e-01   7.7457143e+02   3.5452521e+01   2.6653470e+01   2.0742188e+01   3.7855193e+01   1.7377051e+00   4.6489683e+00   4.0631188e-01   1.7514362e-01   9.3806081e-01   2.6615981e-02   9.7490955e-01   5.8803704e-04   9.9442327e-01   1.4270875e-03   4.8460158e-01   6.0505564e+04   2.7007745e+03   3.6505805e-02   3.6437439e+03   1.1483064e+02   2.7007745e+03
   7.1750000e+02   9.9648727e-01   2.9759468e+01   9.4047875e+01   6.1775983e-01   7.3518750e+02   3.4651579e+01   2.6099378e+01   4.4843750e+01   8.1714898e+01   1.7254692e+00   4.5349143e+00   7.8704665e-01   2.8318753e-01   9.3530052e-01   2.3196722e-02   9.4606531e-01   1.0946779e-03   9.8867727e-01   2.7994246e-03   5.0545095e-01   2.8678163e+04   4.4496181e+03   6.5669307e-02   7.1154780e+03   1.1731060e+02   4.4496181e+03
   6.8788095e+02   9.8347134e-01   2.9110573e+01   9.2629048e+01   6.8093117e-01   7.0523810e+02   3.5477936e+01   2.4391557e+01   1.1285547e+02   1.5815172e+02   1.2355911e+00   2.9868595e+00   2.0642158e+00   9.3544223e-01   9.1864009e-01   3.6750546e-02   8.6696567e-01   3.3028076e-03   9.6931529e-01   8.0868074e-03   5.1504567e-01   9.1111721e+03   8.9188038e+03   1.4242215e-01   1.5968211e+04   6.3788820e+01   8.9188038e+03
   6.7422222e+02   9.8543222e-01   2.8591930e+01   9.0669111e+01   6.4744714e-01   6.9233333e+02   3.4199710e+01   2.4403991e+01   2.3703125e+01   2.4311172e+01   1.6052227e+00   5.3758355e+00   8.8035571e-02   3.5681214e-02   9.1905232e-01   3.2748542e-02   9.9429628e-01   1.1763422e-04   9.9888284e-01   2.5934548e-04   3.0523335e-01   6.5709013e+05   4.4527543e+03   8.0156714e-03   1.8742467e+03   4.1296975e+01   4.4527543e+03
   1.0052407e+03   9.7443263e-01   3.4112740e+01   1.0929981e+02   6.4018090e-01   1.0318704e+03   4.0692089e+01   2.9311109e+01   2.1204297e+02   2.2065584e+02   5.2136559e-01   1.6617670e+00   8.6946962e-01   2.9815221e-01   9.2789019e-01   2.4686356e-02   9.5005182e-01   7.7584675e-04   9.8975427e-01   2.3334156e-03   5.0123391e-01   7.4635241e+04   1.1762444e+04   4.9101848e-02   2.6465468e+04   8.5619708e+01   1.1762444e+04
   6.0856557e+02   1.0037063e+00   2.7187970e+01   8.5598328e+01   6.2959453e-01   6.2317213e+02   3.1887395e+01   2.3662729e+01   2.9001953e+02   3.3515967e+02   6.1323428e-01   1.6689275e+00   1.5674811e+00   5.6603573e-01   9.1531131e-01   3.0534699e-02   9.3119042e-01   1.4461217e-03   9.8484674e-01   3.6192200e-03   5.1806614e-01   4.2002684e+04   1.8977594e+04   7.0042795e-02   3.9947001e+04   9.0259862e+01   1.8977594e+04
   5.5591429e+02   9.9466906e-01   2.6299342e+01   8.3040743e+01   6.7552856e-01   5.6917143e+02   3.1447396e+01   2.2435544e+01   7.6003906e+01   1.1315887e+02   1.2659597e+00   3.0789710e+00   1.2144437e+00   4.8542548e-01   9.0022175e-01   3.9759673e-02   9.0885396e-01   2.3237372e-03   9.7855926e-01   4.4889579e-03   5.5277772e-01   1.2866334e+04   6.4157559e+03   1.1152479e-01   1.4203895e+04   6.7035816e+01   6.4157559e+03
   5.5900000e+02   9.5069402e-01   2.5808408e+01   8.3415857e+01   7.6117243e-01   5.7485714e+02   3.2867097e+01   2.0788224e+01   3.0570312e+01   4.0601591e+01   1.1534103e+00   2.9391452e+00   4.9081493e-01   2.0100325e-01   9.1230880e-01   3.5804817e-02   9.6280919e-01   1.0453527e-03   9.9218910e-01   1.6133891e-03   4.5675747e-01   4.5162934e+04   2.5923309e+03   4.4747812e-02   3.9530656e+03   6.3980500e+01   2.5923309e+03
   4.7550000e+02   1.0085798e+00   2.4175423e+01   7.5731000e+01   6.4931034e-01   4.8625000e+02   2.8662205e+01   2.0845367e+01   2.2289062e+01   3.1578760e+01   1.2249147e+00   3.2573740e+00   3.8426832e-01   1.3785593e-01   8.9308973e-01   3.8180533e-02   9.7270593e-01   7.0535669e-04   9.9407903e-01   1.6838188e-03   4.7482539e-01   5.6320278e+04   2.3027352e+03   3.7394206e-02   3.5295975e+03   6.8575384e+01   2.3027352e+03
   4.7786667e+02   1.0180019e+00   2.4261287e+01   7.5745200e+01   6.1696380e-01   4.8773333e+02   2.8340069e+01   2.1182342e+01   2.8000000e+01   4.2413002e+01   1.2776504e+00   3.1852782e+00   4.7329721e-01   1.8908738e-01   8.8907899e-01   4.4171722e-02   9.6578381e-01   9.8055547e-04   9.9268142e-01   1.5536015e-03   4.9384160e-01   4.6829064e+04   2.6343559e+03   4.4799805e-02   4.5416767e+03   7.3398072e+01   2.6343559e+03
   4.8421429e+02   1.0123921e+00   2.4581857e+01   7.6872857e+01   6.3854573e-01   4.9350000e+02   2.8979436e+01   2.1219499e+01   2.6480469e+01   3.9882863e+01   1.5199740e+00   3.9118076e+00   5.8896283e-01   2.0213193e-01   9.1051926e-01   3.0594428e-02   9.6762440e-01   7.8255690e-04   9.9259193e-01   1.5576108e-03   4.8401297e-01   4.6802714e+04   2.7535837e+03   4.3594925e-02   4.2943991e+03   8.7514545e+01   2.7535837e+03
   4.8116000e+02   9.9657090e-01   2.4513200e+01   7.7510480e+01   6.5549636e-01   4.9428000e+02   2.9347554e+01   2.0997293e+01   4.6988281e+01   7.1815094e+01   1.2661209e+00   3.0387428e+00   8.0198171e-01   2.8080345e-01   8.9616770e-01   3.6186840e-02   9.4288160e-01   1.3620565e-03   9.8781003e-01   3.0735003e-03   4.8666352e-01   2.6854155e+04   3.9891825e+03   6.5908927e-02   6.5640803e+03   6.3523014e+01   3.9891825e+03
   5.8330000e+02   1.0091299e+00   2.6335999e+01   8.2483800e+01   6.4764277e-01   5.9540000e+02   3.0769135e+01   2.2890394e+01   2.2785156e+01   1.7330828e+01   2.1729647e-01   2.6849494e+00   8.7625041e-02   3.7937618e-02   9.4396635e-01   2.4227594e-02   9.9450217e-01   1.2356226e-04   9.9895559e-01   2.6529009e-04   2.9577983e-01   6.6849610e+05   4.4463521e+03   7.8866241e-03   1.7830171e+03   5.8623485e+01   4.4463521e+03
   5.9425000e+02   9.5420544e-01   2.6407277e+01   8.5481250e+01   6.4964508e-01   6.1487500e+02   3.2386909e+01   2.2464292e+01   1.8570312e+01   1.8064887e+01   7.2931864e-01   2.4657301e+00   8.7330326e-02   4.1501656e-02   9.5106273e-01   2.3219628e-02   9.9551560e-01   1.0341389e-04   9.9911742e-01   2.2259921e-04   2.8952140e-01   7.4937540e+05   4.6525384e+03   7.1333269e-03   1.5916907e+03   6.9099405e+01   4.6525384e+03
   5.4885714e+02   9.8943967e-01   2.6157074e+01   8.2616571e+01   6.1856309e-01   5.6400000e+02   3.0389467e+01   2.3080381e+01   3.0015625e+01   6.0826592e+01   2.0337526e+00   5.7429261e+00   6.7472268e-01   2.6540915e-01   9.2302779e-01   3.0158403e-02   9.4759395e-01   1.2981963e-03   9.8874426e-01   2.2978632e-03   5.2379833e-01   2.2744556e+04   3.3845897e+03   7.0211589e-02   5.6069290e+03   1.1123135e+02   3.3845897e+03
   6.1284375e+02   9.7939572e-01   2.7524376e+01   8.7662188e+01   6.6259517e-01   6.3006250e+02   3.3131694e+01   2.3443976e+01   7.6605469e+01   1.5703467e+02   2.1242501e+00   6.2388084e+00   1.8015949e+00   6.8385320e-01   9.2522063e-01   2.8276901e-02   8.6948429e-01   2.9687223e-03   9.7158104e-01   5.1672455e-03   5.4990692e-01   7.1351455e+03   8.1048934e+03   1.5947266e-01   1.4119205e+04   1.1746371e+02   8.1048934e+03
   4.1965000e+02   1.0143470e+00   2.3000407e+01   7.1809800e+01   6.4436506e-01   4.2840000e+02   2.7060078e+01   1.9933119e+01   3.2785156e+01   4.7178143e+01   1.2260809e+00   3.2228425e+00   6.1261319e-01   2.3839353e-01   8.9133587e-01   4.2143657e-02   9.5980963e-01   1.1363761e-03   9.9068409e-01   2.2012576e-03   5.0724161e-01   3.7597169e+04   3.0906815e+03   5.2519622e-02   5.6257154e+03   6.7796841e+01   3.0906815e+03
   5.0086207e+02   9.7332152e-01   2.4911553e+01   7.9500207e+01   7.3881018e-01   5.1327586e+02   3.1155893e+01   2.0360618e+01   5.6738281e+01   9.3354216e+01   1.5204917e+00   3.8201991e+00   1.0861945e+00   4.5191767e-01   9.0815170e-01   3.8099213e-02   9.3130940e-01   1.8901895e-03   9.8458988e-01   3.3364764e-03   4.9855899e-01   2.0537021e+04   5.4155434e+03   7.9173900e-02   8.3037566e+03   6.7377684e+01   5.4155434e+03

];  

class2 = [
    
   7.1406667e+02   9.4554152e-01   2.9608493e+01   9.6061200e+01   7.7291657e-01   7.3120000e+02   3.8394954e+01   2.3308596e+01   4.1839844e+01   5.8908518e+01   1.0639428e+00   2.5916190e+00   7.9364516e-01   3.3005708e-01   9.1588235e-01   3.4853742e-02   9.4955551e-01   1.2206511e-03   9.8829131e-01   3.0034054e-03   5.3936309e-01   3.0059492e+04   3.5968244e+03   6.4930103e-02   7.8579067e+03   8.6110922e+01   3.5968244e+03
   7.4037500e+02   9.5870937e-01   3.0241528e+01   9.7206625e+01   7.5010075e-01   7.5612500e+02   3.8020366e+01   2.4496621e+01   4.6273438e+01   6.5703996e+01   1.1924846e+00   2.9280621e+00   8.2752643e-01   2.9011714e-01   9.1775265e-01   2.8700966e-02   9.4436100e-01   1.1209486e-03   9.8742488e-01   2.7940129e-03   5.2359383e-01   2.7735649e+04   3.8239540e+03   6.8822790e-02   7.9035012e+03   8.2428445e+01   3.8239540e+03
   4.9648000e+02   9.9918179e-01   2.4721856e+01   7.7981280e+01   6.3463262e-01   5.0912000e+02   2.9313964e+01   2.1385230e+01   4.8484375e+01   8.1422643e+01   1.6882617e+00   4.5569242e+00   1.2378531e+00   5.0657143e-01   9.1815997e-01   3.3366775e-02   9.1583588e-01   2.2735137e-03   9.8011207e-01   3.8564238e-03   5.0843325e-01   1.4237347e+04   5.0248071e+03   1.0034831e-01   7.6311494e+03   6.4439615e+01   5.0248071e+03
   5.7425000e+02   9.8396254e-01   2.6362627e+01   8.3885750e+01   7.0802551e-01   5.9362500e+02   3.2198523e+01   2.1979795e+01   1.7945312e+01   3.2622694e+01   1.7771348e+00   4.8924586e+00   3.4995652e-01   1.4769840e-01   9.1067365e-01   3.7563485e-02   9.6853906e-01   8.5961009e-04   9.9314854e-01   1.3167775e-03   4.9537908e-01   3.8020807e+04   2.1382345e+03   4.6044922e-02   3.3001064e+03   7.7596960e+01   2.1382345e+03
   5.0471429e+02   1.0139985e+00   2.5003156e+01   7.8392214e+01   6.3897528e-01   5.1492857e+02   2.9688887e+01   2.1454719e+01   5.5203125e+01   8.9902026e+01   1.5793810e+00   4.0646949e+00   9.2320406e-01   4.0434223e-01   9.0637675e-01   4.0888807e-02   9.3321731e-01   1.9339780e-03   9.8589430e-01   3.0991588e-03   4.9054534e-01   2.2763335e+04   4.6344777e+03   7.5520833e-02   7.6770538e+03   7.1655472e+01   4.6344777e+03
   5.2942857e+02   9.8802511e-01   2.5558118e+01   8.1155143e+01   7.0270063e-01   5.4209524e+02   3.1375697e+01   2.1251160e+01   4.3429688e+01   8.2791748e+01   1.9369164e+00   5.3434236e+00   4.3582378e-01   2.0991009e-01   9.1681698e-01   3.9958614e-02   9.4730849e-01   1.6378317e-03   9.9025333e-01   2.8061472e-03   3.9915429e-01   3.4699738e+04   3.9815410e+03   5.3498445e-02   3.8037455e+03   4.5764979e+01   3.9815410e+03
   4.4689474e+02   1.0149679e+00   2.3708543e+01   7.4226526e+01   6.4147833e-01   4.5810526e+02   2.8060249e+01   2.0432922e+01   3.3167969e+01   5.1587724e+01   1.3020614e+00   3.1564871e+00   7.0123692e-01   2.4603342e-01   9.0351734e-01   3.3660802e-02   9.4180682e-01   1.4545991e-03   9.8589731e-01   2.6545378e-03   5.5600797e-01   1.8474452e+04   3.2082572e+03   8.0605469e-02   7.1711153e+03   7.2819603e+01   3.2082572e+03
   4.5370000e+02   9.8222953e-01   2.3887591e+01   7.6025700e+01   7.1360398e-01   4.6535000e+02   2.9854002e+01   1.9544370e+01   3.5445312e+01   5.3270867e+01   1.2519132e+00   3.0202323e+00   7.7266847e-01   3.8574951e-01   8.9858324e-01   5.0421815e-02   9.3784927e-01   2.1926559e-03   9.8486335e-01   3.6287736e-03   5.6847364e-01   1.6895246e+04   3.2559020e+03   8.7122396e-02   8.0939301e+03   7.2613847e+01   3.2559020e+03
   5.2052632e+02   9.8191768e-01   2.5475817e+01   8.1086737e+01   6.9958254e-01   5.3389474e+02   3.1549738e+01   2.1044039e+01   3.8632812e+01   6.8651992e+01   1.6727099e+00   4.3441555e+00   7.9684784e-01   3.6376777e-01   9.1019128e-01   4.0860195e-02   9.3265523e-01   2.0398039e-03   9.8474585e-01   3.4202561e-03   5.7248400e-01   1.5686211e+04   3.7160558e+03   9.3274740e-02   8.8043257e+03   1.0020419e+02   3.7160558e+03
   5.6042857e+02   9.8050637e-01   2.6461731e+01   8.4093857e+01   6.8704453e-01   5.7371429e+02   3.2204753e+01   2.2223493e+01   3.0648438e+01   5.7424404e+01   1.7830058e+00   4.7272102e+00   6.6367132e-01   2.8095982e-01   9.2129086e-01   3.3223192e-02   9.6270129e-01   1.0102512e-03   9.9170162e-01   1.8232428e-03   4.8852123e-01   4.0727836e+04   3.5668501e+03   4.9334491e-02   5.0083027e+03   1.0818310e+02   3.5668501e+03
   6.2622222e+02   1.0218134e+00   2.7585422e+01   8.5739111e+01   5.8308290e-01   6.3811111e+02   3.1114262e+01   2.4802955e+01   2.2015625e+01   3.5763380e+01   1.6010132e+00   4.3822164e+00   4.7501147e-01   1.6464154e-01   9.2882597e-01   2.4552918e-02   9.7326451e-01   5.5232761e-04   9.9388310e-01   1.3483918e-03   4.7365152e-01   5.8812176e+04   2.6384485e+03   3.7358037e-02   3.5591612e+03   9.9760378e+01   2.6384485e+03
   6.5568182e+02   1.0191729e+00   2.8422926e+01   8.8659182e+01   6.1805545e-01   6.6913636e+02   3.2838908e+01   2.4979527e+01   5.6347656e+01   8.1237924e+01   1.2810014e+00   3.1635608e+00   6.5216870e-01   2.8122332e-01   8.9627627e-01   4.4595316e-02   9.3234779e-01   1.7005320e-03   9.8459173e-01   3.2136249e-03   5.7201305e-01   1.9613722e+04   4.3108810e+03   8.8293005e-02   1.1968163e+04   6.1380767e+01   4.3108810e+03
   5.6131579e+02   1.0262340e+00   2.6522991e+01   8.2359053e+01   5.9572768e-01   5.7178947e+02   3.0402874e+01   2.3461825e+01   4.1660156e+01   7.2029281e+01   1.8122785e+00   5.1662265e+00   6.8166852e-01   3.0182354e-01   9.0967389e-01   3.9878514e-02   9.4953877e-01   1.4280953e-03   9.8946003e-01   2.4469980e-03   4.7687801e-01   3.1330046e+04   3.8862662e+03   5.7906539e-02   5.5454842e+03   6.6651702e+01   3.8862662e+03
   6.6976000e+02   9.8974844e-01   2.8820603e+01   9.1233680e+01   6.5800871e-01   6.8640000e+02   3.4110420e+01   2.4777197e+01   6.5406250e+01   1.1134363e+02   1.6053298e+00   4.1179852e+00   1.3353668e+00   5.4582821e-01   9.3124808e-01   2.8015434e-02   9.2168324e-01   1.8503812e-03   9.8161175e-01   3.6765871e-03   5.5058574e-01   1.6606330e+04   6.6028488e+03   9.8295989e-02   1.2534122e+04   1.1210107e+02   6.6028488e+03
   6.2276190e+02   9.9867254e-01   2.7841559e+01   8.7787619e+01   5.7082796e-01   6.3861905e+02   3.2056815e+01   2.4749086e+01   5.1085938e+01   8.4993612e+01   1.5541364e+00   3.9919463e+00   1.3193378e+00   4.8472729e-01   9.2800178e-01   2.6359758e-02   9.3844466e-01   1.3243517e-03   9.8443254e-01   2.9071910e-03   5.6493869e-01   2.2635425e+04   5.4956247e+03   8.2329644e-02   1.0963037e+04   1.1649915e+02   5.4956247e+03
   7.1534615e+02   9.5414847e-01   2.9957007e+01   9.6483846e+01   7.4762862e-01   7.3250000e+02   3.7957492e+01   2.4117220e+01   7.2652344e+01   1.2762067e+02   1.6295807e+00   4.2386059e+00   1.4483742e+00   7.5372511e-01   9.2760493e-01   3.7561074e-02   9.1326289e-01   2.4899684e-03   9.8086181e-01   5.5609536e-03   5.2365657e-01   1.4629028e+04   6.8327019e+03   1.0178177e-01   1.1831946e+04   1.1173175e+02   6.8327019e+03
   5.1525000e+02   1.0253218e+00   2.5335909e+01   7.8515500e+01   5.1858092e-01   5.2425000e+02   2.8293278e+01   2.3094256e+01   8.0507812e+00   1.1441298e+01   1.4512762e+00   4.1597704e+00   5.0826668e-02   1.9490794e-02   9.1359857e-01   3.3082049e-02   9.9804787e-01   4.4900950e-05   9.9958357e-01   1.2920586e-04   1.8879278e-01   1.0921264e+06   5.4164656e+03   5.0891097e-03   9.4482329e+02   4.1062999e+01   5.4164656e+03
   1.0173000e+03   9.5898958e-01   3.5600831e+01   1.1478740e+02   7.0789910e-01   1.0412000e+03   4.4209911e+01   2.9211443e+01   3.9738281e+01   3.9177898e+01   3.6513726e-01   1.5394146e+00   1.4717626e-01   6.1927476e-02   8.9540132e-01   4.3952335e-02   9.9052643e-01   1.8085533e-04   9.9797169e-01   5.6243513e-04   4.5591789e-01   3.9579447e+05   4.0064242e+03   1.3066379e-02   5.8348565e+03   5.9975463e+01   4.0064242e+03
   5.1056098e+02   1.0121772e+00   2.5169590e+01   7.9049707e+01   6.3716197e-01   5.2126829e+02   3.0077161e+01   2.1472801e+01   8.1769531e+01   1.4181625e+02   1.6940860e+00   4.6242937e+00   1.7573954e+00   8.1198837e-01   9.1866542e-01   3.7454413e-02   9.0193502e-01   2.8721556e-03   9.7850541e-01   6.1616854e-03   5.3906745e-01   1.2028615e+04   7.3443249e+03   1.1783176e-01   1.4394944e+04   1.2374114e+02   7.3443249e+03
   1.1113000e+03   9.2859011e-01   3.7453395e+01   1.2226700e+02   6.6860901e-01   1.1478000e+03   4.5675426e+01   3.1597809e+01   4.3410156e+01   5.7163952e+01   1.0171824e+00   2.4662142e+00   1.6712151e-01   7.3584704e-02   9.2961573e-01   3.0940093e-02   9.8966579e-01   2.0554249e-04   9.9785928e-01   6.0491659e-04   4.0281841e-01   3.9644175e+05   4.6956852e+03   1.2954747e-02   4.6856510e+03   6.8194954e+01   4.6956852e+03
   7.5725000e+02   9.6431404e-01   3.0705748e+01   9.8379375e+01   6.6790376e-01   7.7850000e+02   3.6947927e+01   2.6194418e+01   4.7328125e+01   4.2045934e+01   8.2288285e-01   2.5360995e+00   2.3455266e-01   1.0410092e-01   9.1826991e-01   3.6226584e-02   9.8864852e-01   2.6801896e-04   9.9758613e-01   6.5816417e-04   3.9066402e-01   3.7921518e+05   4.4940662e+03   1.3440572e-02   4.5521116e+03   5.9132957e+01   4.4940662e+03
   6.2300000e+02   9.3073921e-01   2.7871015e+01   9.1168833e+01   7.6189422e-01   6.4491667e+02   3.6013435e+01   2.2175288e+01   2.9203125e+01   3.1816227e+01   7.1574499e-01   2.0826054e+00   1.4424057e-01   8.1573969e-02   8.9029476e-01   6.1964109e-02   9.9294469e-01   2.3532307e-04   9.9832298e-01   5.8933110e-04   4.3742737e-01   4.6194996e+05   4.2323212e+03   1.1173983e-02   4.6399942e+03   6.3588235e+01   4.2323212e+03
   8.1722222e+02   1.0136543e+00   3.1986531e+01   1.0001156e+02   5.3117422e-01   8.3800000e+02   3.5553909e+01   2.9200351e+01   2.8730469e+01   4.8825998e+01   1.5704874e+00   4.0198332e+00   5.4947879e-01   2.0476911e-01   9.3706637e-01   2.3349985e-02   9.6537443e-01   6.8008359e-04   9.9146460e-01   1.7149792e-03   5.5397071e-01   4.2661727e+04   3.2932861e+03   5.0837764e-02   6.5052778e+03   1.1606416e+02   3.2932861e+03
   5.2418182e+02   1.0088990e+00   2.5600208e+01   8.0308182e+01   6.2837854e-01   5.3536364e+02   3.0490026e+01   2.1877833e+01   2.2523438e+01   3.3514402e+01   1.3414767e+00   3.4308415e+00   3.7966519e-01   1.8072250e-01   8.9328501e-01   5.0628789e-02   9.7248135e-01   8.0838933e-04   9.9383374e-01   1.8061357e-03   4.7850328e-01   5.6201460e+04   2.3752975e+03   3.7862142e-02   3.6339195e+03   6.8943519e+01   2.3752975e+03
   7.3114286e+02   1.0005230e+00   3.0073831e+01   9.4554571e+01   6.2145073e-01   7.4647619e+02   3.4926190e+01   2.6370319e+01   5.9976562e+01   1.0872844e+02   1.8188228e+00   5.0812524e+00   1.3445110e+00   5.5179376e-01   9.3501708e-01   2.6562233e-02   9.2824743e-01   1.5807480e-03   9.8274929e-01   3.8795853e-03   5.5558105e-01   1.9220466e+04   6.5966175e+03   9.1503002e-02   1.1847712e+04   1.2033764e+02   6.5966175e+03
   4.8510000e+02   1.0103316e+00   2.4578024e+01   7.6899000e+01   6.0259042e-01   4.9770000e+02   2.8222008e+01   2.1836514e+01   1.8949219e+01   2.7738297e+01   1.3439623e+00   3.4339328e+00   3.6530130e-01   1.4815102e-01   9.0918462e-01   3.6686197e-02   9.7678004e-01   6.1396616e-04   9.9446809e-01   1.2714957e-03   4.8183376e-01   6.3168930e+04   2.3042960e+03   3.4744828e-02   3.3968271e+03   7.2420364e+01   2.3042960e+03
   7.7152632e+02   9.7088489e-01   3.0420161e+01   9.7312632e+01   6.4912592e-01   7.9668421e+02   3.6050488e+01   2.6336307e+01   5.7261719e+01   4.3368438e+01   5.6078445e-01   4.0887693e+00   3.6325557e-01   1.3220848e-01   9.3279778e-01   2.4419615e-02   9.8629447e-01   2.6105276e-04   9.9682273e-01   5.9677701e-04   4.7668825e-01   2.9655641e+05   4.6033835e+03   1.6921725e-02   8.1534667e+03   8.0026837e+01   4.6033835e+03
   7.0692500e+02   9.7158595e-01   2.9486007e+01   9.4231200e+01   6.5139988e-01   7.2675000e+02   3.5225359e+01   2.5293622e+01   1.1045703e+02   1.0094397e+02   3.9141421e-01   1.6601779e+00   8.3052494e-01   3.3910761e-01   9.1749191e-01   3.3641433e-02   9.7357217e-01   5.8731399e-04   9.9337356e-01   1.5034830e-03   5.2826256e-01   1.3091473e+05   7.6199076e+03   3.1760831e-02   1.8674913e+04   9.8126139e+01   7.6199076e+03
   5.3992000e+02   9.9723107e-01   2.5598954e+01   8.0869360e+01   6.8113870e-01   5.5212000e+02   3.1071827e+01   2.1460374e+01   5.2726562e+01   4.4909620e+01   2.3673150e-01   1.6429278e+00   2.5477148e-01   1.1646512e-01   9.2182150e-01   3.5694704e-02   9.8725926e-01   3.4504265e-04   9.9715231e-01   7.9242232e-04   4.4206839e-01   3.2043388e+05   4.8318610e+03   1.5880416e-02   6.6645061e+03   7.1109268e+01   4.8318610e+03
   7.5400000e+02   9.5644827e-01   3.0287098e+01   9.7355857e+01   6.9760234e-01   7.7442857e+02   3.6782300e+01   2.5510481e+01   4.1234375e+01   4.2473426e+01   6.9074524e-01   2.1443900e+00   2.1925784e-01   1.1736304e-01   9.3325912e-01   3.5686289e-02   9.9011025e-01   2.7002388e-04   9.9754822e-01   7.2651316e-04   4.5255673e-01   3.5135708e+05   4.6463813e+03   1.3979535e-02   6.1650125e+03   7.4201968e+01   4.6463813e+03
   4.5331250e+02   9.8448330e-01   2.3885061e+01   7.5950500e+01   7.0587519e-01   4.6381250e+02   2.9790593e+01   1.9642951e+01   2.8332031e+01   3.7780313e+01   8.9332355e-01   2.2362511e+00   5.5948016e-01   2.2927814e-01   8.9530567e-01   4.2774747e-02   9.6526539e-01   9.9423036e-04   9.9204830e-01   1.6356915e-03   4.8938387e-01   4.3195991e+04   2.6842209e+03   4.6217177e-02   4.6409745e+03   7.1417119e+01   2.6842209e+03
   3.6413333e+02   9.9813893e-01   2.1491813e+01   6.7738533e+01   7.1954067e-01   3.7126667e+02   2.6749661e+01   1.7636359e+01   2.1335938e+01   3.2009013e+01   1.3172507e+00   3.3672639e+00   3.5277715e-01   1.4289214e-01   8.6519952e-01   5.4423240e-02   9.7360125e-01   8.5330928e-04   9.9423968e-01   1.3716003e-03   4.7354218e-01   5.7415300e+04   2.3667352e+03   3.6919488e-02   3.4536636e+03   5.7569802e+01   2.3667352e+03
   6.2920833e+02   1.0237754e+00   2.7756097e+01   8.6728000e+01   6.2892756e-01   6.4212500e+02   3.2547200e+01   2.4002483e+01   5.8988281e+01   8.8815450e+01   1.2597731e+00   3.0716835e+00   7.5682329e-01   3.0317940e-01   9.0523420e-01   3.7808149e-02   9.2918087e-01   1.6501888e-03   9.8553447e-01   3.8443139e-03   5.0372136e-01   2.1102412e+04   4.3346039e+03   7.9992224e-02   8.5529414e+03   6.6600718e+01   4.3346039e+03

]; % microscopy images of brain tissue LowGrade

classLabels =[ones(1,size(class1,1)) (-1)*ones(1,size(class2,1))]';
superClass=[class1;class2];%form superclass
% superClass=mapstd(superClass')';%Normalize data to 0 mean
[superClass, C1, S1] = normalize(superClass);
[class1,class2]=classConstruction(superClass,classLabels );%reconstruct 2 classes to feed algorithms

% sums for majority voting of classification
ls = 0;
hs = 0;

%----------KNN (3)-------------

% class feature selection for KNN
% class1
knn_cl1(:,1) = class1(:,2); % choose specific features from classes
knn_cl1(:,2) = class1(:,5);
knn_cl1(:,3) = class1(:,13);
knn_cl1(:,4) = class1(:,22);

% class2
knn_cl2(:,1) = class2(:,2);
knn_cl2(:,2) = class2(:,5);
knn_cl2(:,3) = class2(:,13);
knn_cl2(:,4) = class2(:,22);

% feature selection for KNN
knn_x(1,1) = f(1,2);
knn_x(1,2) = f(1,5);
knn_x(1,3) = f(1,13);
knn_x(1,4) = f(1,22);

% mean value for KNN 
knn_m(1,1) = C1(1,2);
knn_m(1,2) = C1(1,5);
knn_m(1,3) = C1(1,13);
knn_m(1,4) = C1(1,22);

% std for KNN
knn_std(1,1) = S1(1,2);
knn_std(1,2) = S1(1,5);
knn_std(1,3) = S1(1,13);
knn_std(1,4) = S1(1,22);

% normalize the current image pattern
knn_x = (knn_x-knn_m)./knn_std;

% KNN Classification
[knn_class]=Classifiers.KNN_classifier(knn_x,knn_cl1,knn_cl2,3);

if knn_class == 1
    fprintf("KNN(3): High Grade")
    hs = hs + 1;
elseif knn_class == 2
    fprintf("KNN(3): Low Grade")
    ls = ls + 1;
end


%---------------PNN--------------------

pnn_cl1(:,1) = class1(:,22);
pnn_cl1(:,2) = class1(:,25);

pnn_cl2(:,1) = class2(:,22);
pnn_cl2(:,2) = class2(:,25);

pnn_x(1,1) = f(1,22);
pnn_x(1,2) = f(1,25);


pnn_m(1,1) = C1(1,22);
pnn_m(1,2) = C1(1,25);

pnn_std(1,1) = S1(1,22);
pnn_std(1,2) = S1(1,25);


pnn_x = (pnn_x-pnn_m)./pnn_std;

[pnn_class] = Classifiers.PNN_classifier(pnn_x,pnn_cl1,pnn_cl2);


if pnn_class == 1
    fprintf("PNN:    High Grade")
    hs = hs + 1;
elseif pnn_class == 2
    fprintf("PNN:    Low Grade")
    ls = ls + 1;
end

%-------------Bayesian-----------------

b_cl1(:,1) = class1(:, 17);
b_cl1(:,2) = class1(:, 23);
b_cl1(:,3) = class1(:, 25);
b_cl1(:,4) = class1(:, 27);

b_cl2(:,1) = class2(:, 17);
b_cl2(:,2) = class2(:, 23);
b_cl2(:,3) = class2(:, 25);
b_cl2(:,4) = class2(:, 27);

b_x(1,1) = f(1,17);
b_x(1,2) = f(1,23);
b_x(1,3) = f(1,25);
b_x(1,4) = f(1,27);

b_m(1,1) = C1(1,17);
b_m(1,2) = C1(1,23);
b_m(1,3) = C1(1,25);
b_m(1,4) = C1(1,27);

b_std(1,1) = S1(1,17);
b_std(1,2) = S1(1,23);
b_std(1,3) = S1(1,25);
b_std(1,4) = S1(1,27);

b_x = (b_x-b_m)./b_std;

[b_class] = Classifiers.Bayesian_classifier(b_x,b_cl1,b_cl2);

if b_class == 1
    fprintf("Bayes:  High Grade")
    hs = hs + 1;
elseif b_class == 2
    fprintf("Bayes:  Low Grade")
    ls = ls + 1;
end


%% Final Result (Majority Voting)

if hs>ls
    fprintf("=======FINAL RESULT========")
    fprintf("\nImage is: High Grade\n")
elseif ls>hs
    fprintf("=======FINAL RESULT========")
    fprintf("\nImage is: Low Grade\n")
end



