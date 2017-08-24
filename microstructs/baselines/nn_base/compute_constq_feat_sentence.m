% Compute constant-Q features for sentence segments from TIMIT.

clear;
close all

is_plot = 0; % plot

% Const-Q setting
addpath('~/Documents/ProJEX/withRita/EyBreath/CQT_toolbox_2013');
fs = 16000;
fmin = 20;
fmax = fs/2;
B = 48;

% File path
wavpath = '~/Downloads/misc_tmp/sentence_segments';
savepath = '~/Downloads/misc_tmp/sentence_constq_feats';
if ~(exist(savepath, 'dir'))
    mkdir(savepath);
end

% Wav list
fid = fopen(fullfile(wavpath, 'all_sentence_wavs.ctl'));
raw = textscan(fid, '%s');
phnlist = raw{1};
fclose(fid);

num_f = length(phnlist);

fprintf('Extracting features for wavs in %s\n', wavpath);
tic;
parfor idx = 1:num_f
    fpath = phnlist{idx};
    fprintf('%s\n', fpath);
    % Read wav
    wav = audioread(fullfile(wavpath, fpath));
    x = wav(:);
    xlen = length(x);
    % Compute const-q
    Xcq = cqt(x, B, fs, fmin, fmax);
    c = Xcq.c;
    if is_plot
        figure;
        subplot(131);
        plot(x);
        subplot(132);
        imagesc(flipud(abs(c)));
        subplot(133);
        imagesc(20*log10(abs(flipud(c))+eps)); % in log scale
        pause;
    end
    % write coefficients
    [base,fn,~] = fileparts(fpath);
    sfpath = fullfile(savepath, base);
    if ~(exist(sfpath, 'dir'))
        mkdir(sfpath);
    end
        
    csvwrite(fullfile(sfpath, strcat(fn, '.constq')), abs(c));
end
toc;
disp('Done!');