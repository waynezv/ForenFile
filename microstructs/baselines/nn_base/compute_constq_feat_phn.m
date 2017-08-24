% Compute constant-Q features for phonemes extracted from TIMIT.

clear;
close all

plot = 0; % plot

% Const-Q setting
addpath('~/Documents/ProJEX/withRita/EyBreath/CQT_toolbox_2013');
fs = 16000;
fmin = 20;
fmax = fs/2;
B = 48;

% File path
wavpath = '~/Downloads/misc_tmp/phsegwav_out_samples';
savepath = '~/Downloads/misc_tmp/phsegwav_out_constq_feats';
if ~(exist(savepath, 'dir'))
    mkdir(savepath);
end

% Wav list
fid = fopen(fullfile(wavpath, 'timit_phsegwav_test.ctl'));
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
    wav = audioread(fullfile(wavpath, strcat(fpath, '.wav')));
    x = wav(:);
    xlen = length(x);
    % Compute const-q
    Xcq = cqt(x, B, fs, fmin, fmax);
    c = Xcq.c;
    if plot
        figure;
        subplot(121);
        imagesc(flipud(abs(c)));
        subplot(122);
        imagesc(20*log10(abs(flipud(c))+eps)); % in log scale
        xlabel('time', 'FontSize', 12, 'Interpreter','latex');
        ylabel('frequency', 'FontSize', 12, 'Interpreter','latex');
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