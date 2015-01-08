function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

IMAGES = loadMNISTImages('train-images.idx3-ubyte');

patchsize = 28;  % we'll use 28x28 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 

% % retrieve the width and height of the images:
ni = size(IMAGES,2)
nf = size(IMAGES,1)

patches = IMAGES(:,1:numpatches);

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
% patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
