% Convert spectrograms to wavs

close all;
clear all;

image_dir = '~/Downloads/misc_tmp/ganae_results/09151245/pngs';
addpath(genpath(image_dir));

test_img = 'fake_samples_epoch_199.png';
img = imread(fullfile(image_dir, test_img));
img = im2double(img);
img = imresize(img, [834, 3616]);

specs = {};
for i = 1:2
    for j = 1:8
        specs{i,j} = ...
            img((i-1)*417+1 : i*417, (j-1)*452+1 : j*452);
    end
end

test_spec = flipud(specs{1,1});
y = test_spec;

fs = 16000;
xLngth = 32000;
wndLngth = 512;
noverlap = wndLngth*3/4;
hp = wndLngth - noverlap;
nfft = 512;

% GL-method
% wav = gl_spec2wav(y, 'zero_phase', xLngth, wndLngth, noverlap, hp, nfft);

