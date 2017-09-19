function wav = gl_spec2wav(spec, phase_method, ...
    xLngth, wndLngth, noverlap, hp, nfft)

y = spec;
y = [y; conj(flipud(y(2:nfft/2,:)))];

og_mag = abs(y);
og_xmat = ifft(y);
og_xmat = og_xmat(1:wndLngth, :);

noWnds = size(og_xmat,2);

if strcmp(phase_method, 'zero_phase')
    %% Testing for zero phase as the initial phase
    zero_xmat = ifft(og_mag);
    %     zero_xmat = circshift(zero_xmat, nfft/2);
    %     zero_xmat = zero_xmat((nfft-wndLngth)/2+1:(nfft+wndLngth)/2, :);
    
    out_zero = zeros(hp*(noWnds-1) + wndLngth, 1);
    for i = 1:noWnds
        out_zero((i-1)*hp+1:(i-1)*hp+wndLngth) = out_zero((i-1)*hp+1:(i-1)*hp+wndLngth) + zero_xmat(:,i);
    end
    
    % starting phase
    sum_og_mag = sum(og_mag(:));
    
    for j = 1:100
        out_stft = spectrogram(out_zero, hamming(wndLngth), noverlap, nfft);
        out_stft = [out_stft; conj(flipud(out_stft(2:nfft/2,:)))];
        out_stft_phse = out_stft./abs(out_stft);
        
        err(j) = sum((og_mag(:) - abs(out_stft(:))).^2)/sum_og_mag;
        
        zero_xmat = ifft(og_mag.*out_stft_phse);
        zero_xmat = zero_xmat((nfft-wndLngth)/2+1:(nfft+wndLngth)/2, :);
        
        out_zero = zeros(hp*(noWnds-1) + wndLngth, 1);
        for i = 1:noWnds
            out_zero((i-1)*hp+1:(i-1)*hp+wndLngth) = out_zero((i-1)*hp+1:(i-1)*hp+wndLngth) + zero_xmat(:,i);
        end
    end
    
elseif strcmp(phase_method, 'random_phase')
    %% Testing with random phase as the initial condition
    noise = wgn(xLngth, 1, 0);
    
    noise_spect = spectrogram(noise, hamming(wndLngth), noverlap, nfft);
    noise_spect = [noise_spect; conj(flipud(noise_spect(2:nfft/2,:)))];
    abs_noise = abs(noise_spect);
    phse_noise = noise_spect./abs_noise;
    
    rndm_xmat = ifft(og_mag.*phse_noise);
    %     rndm_xmat = circshift(rndm_xmat, nfft/2);
    %     rndm_xmat = rndm_xmat((nfft-wndLngth)/2+1:(nfft+wndLngth)/2, :);
    
    out_rndm = zeros(hp*(noWnds-1) + wndLngth, 1);
    for i = 1:noWnds
        out_rndm((i-1)*hp+1:(i-1)*hp+wndLngth) = out_rndm((i-1)*hp+1:(i-1)*hp+wndLngth) + rndm_xmat(:,i);
    end
    
    sum_og_mag = sum(og_mag(:));
    
    for j = 1:100
        out_stft = spectrogram(out_rndm, hamming(wndLngth), noverlap, nfft);
        out_stft = [out_stft; conj(flipud(out_stft(2:nfft/2,:)))];
        out_stft_phse = out_stft./abs(out_stft);
        
        err(j) = sum((og_mag(:) - abs(out_stft(:))).^2)/sum_og_mag;
        
        rndm_xmat = ifft(og_mag.*out_stft_phse);
        rndm_xmat = rndm_xmat((nfft-wndLngth)/2+1:(nfft+wndLngth)/2, :);
        
        out_rndm = zeros(hp*(noWnds-1) + wndLngth, 1);
        for i = 1:noWnds
            out_rndm((i-1)*hp+1:(i-1)*hp+wndLngth) = out_rndm((i-1)*hp+1:(i-1)*hp+wndLngth) + rndm_xmat(:,i);
        end
    end
end

plot(err);
end