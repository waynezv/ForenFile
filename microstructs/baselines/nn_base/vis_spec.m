addpath('~/Documents/ProJEX/withRita/EyBreath/CQT_toolbox_2013');
fs = 16000;
fmin = 20;
fmax = fs/2;
B = 48;

path = '~/Downloads/misc_tmp';
fn = 'OW/55';

% Spec
spec = csvread(fullfile(path, 'phsegwav_feat_samples/train', fn));
figure;
subplot(121);
imagesc(20*log10(abs(flipud(spec))+eps));
xlabel('time', 'FontSize', 12, 'Interpreter','latex');
ylabel('frequency', 'FontSize', 12, 'Interpreter','latex');

% Const-q
wav = audioread(fullfile(path, 'phsegwav_out_samples/train', ...
    strcat(fn, '.wav')));
x = wav(:);
xlen = length(x);

Xcq = cqt(x, B, fs, fmin, fmax);
c = Xcq.c;

subplot(122);
imagesc(20*log10(abs(flipud(c))+eps));
xlabel('time', 'FontSize', 12, 'Interpreter','latex');
ylabel('frequency', 'FontSize', 12, 'Interpreter','latex');