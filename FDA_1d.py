
import torch
import pywt
import numpy as np 

def bandwidth_freq_mutate_1d_with_fs(src, trg, fs=1000, cutoff_freq_lower=10,cutoff_freq_upper=100):
    """ 
    Args:
        src: 源幅度谱 (B, C, L)
        trg: 目标幅度谱 
        fs: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
    """
    t = src.shape[-1]
    nyquist = fs / 2  # Nyquist频率
    total_freq_bins = t  # FFT后的频点总数（rfft的特殊性需处理）
    # print(B,C,t)
    # 计算截止频率对应的频点位置
    cutoff_bin_lower = int( (cutoff_freq_lower / nyquist) * t )  # 只考虑单侧频谱
    cutoff_bin_upper = int( (cutoff_freq_upper / nyquist) * t )  # 只考虑单侧频谱
    # print(cutoff_bin)
    
    # 替换高频区域（考虑rfft的对称性）
    src[..., cutoff_bin_lower:cutoff_bin_upper] = trg[..., cutoff_bin_lower:cutoff_bin_upper]
    return src
def high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=1000, cutoff_freq=300):
    """ 使用物理频率控制低频区域
    Args:
        amp_src: 源幅度谱 (B, C, L)
        amp_trg: 目标幅度谱 
        fs: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
    """
    t = amp_src.shape[-1]
    nyquist = fs / 2  # Nyquist频率
    total_freq_bins = t  # FFT后的频点总数（rfft的特殊性需处理）
    # print(B,C,t)
    # 计算截止频率对应的频点位置
    cutoff_bin = int( (cutoff_freq / nyquist) * t )  # 只考虑单侧频谱
    # print(cutoff_bin)
    
    # 替换高频区域（考虑rfft的对称性）
    amp_src[..., cutoff_bin:] = amp_trg[..., cutoff_bin:]
    return amp_src

def FDA_1d_with_fs(src_signal, trg_signal, fs=1000, cutoff_freq=300,cutoff_freq_upper=None):
    # 输入形状: (B, C, T)
    src = src_signal
    trg = trg_signal

    # 计算RFFT（实数信号FFT）
    fft_src = torch.fft.rfft(src, dim=-1, norm='forward')  # 输出形状 (B, C, L+1)
    fft_trg = torch.fft.rfft(trg, dim=-1, norm='forward')
    # print(fft_src.shape)

    # rfreqs = torch.fft.rfftfreq(1800, 1/1000)  # 得到非负频率
    # # print(src.shape)
    # print(rfreqs)
    # exit(0)
    
    # 分解幅度和相位
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg,pha_trg = torch.abs(fft_trg),torch.angle(fft_trg)
    
    # 使用物理频率进行高频替换
    # amp_src_mutated = high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=fs, cutoff_freq=cutoff_freq)
    if cutoff_freq_upper==None:
        pha_src_mutated = high_freq_mutate_1d_with_fs(pha_src, pha_trg, fs=fs, cutoff_freq=cutoff_freq)
    else:
        pha_src_mutated=bandwidth_freq_mutate_1d_with_fs(pha_src, pha_trg,fs=fs,cutoff_freq_lower=cutoff_freq,cutoff_freq_upper=cutoff_freq_upper)
        # amp_src_mutated=bandwidth_freq_mutate_1d_with_fs(amp_src, amp_trg,fs=fs,cutoff_freq_lower=cutoff_freq,cutoff_freq_upper=cutoff_freq_upper)
    
    # 重建信号
    fft_mixed = torch.polar(amp_src, pha_src_mutated)
    # fft_mixed = torch.polar(amp_src_mutated, pha_src)
    mixed = torch.fft.irfft(fft_mixed, n=src.size(-1), dim=-1, norm='forward')
    
    return mixed.to(src_signal.dtype)






def cwt_transform(signal, wavelet='morl', scales=64):
    """
    计算输入信号的连续小波变换（CWT）
    - signal: (B, C, T) 输入信号
    - wavelet: 选择的小波基函数（默认 morl）
    - scales: 变换的尺度数量
    返回：
    - 小波系数 (B, C, scales, T)
    - 频率索引
    """
    print(signal.shape)
    B, C, T = signal.shape
    scales_arr = np.arange(1, scales + 1)  # 选择尺度
    coeffs_list = []

    for b in range(B):
        batch_coeffs = []
        for c in range(C):
            coeffs, freqs = pywt.cwt(signal[b, c].cpu().numpy(), scales_arr, wavelet)  # (scales, T)
            batch_coeffs.append(coeffs)
        coeffs_list.append(batch_coeffs)

    cwt_coeffs = torch.tensor(coeffs_list)  # (B, C, scales, T)
    return cwt_coeffs

def icwt(coeffs, scales, wavelet, dt=1.0):
    """
    逆连续小波变换 (Inverse Continuous Wavelet Transform, ICWT)
    
    Args:
        coeffs: 小波变换系数 (B, C, Scales, T)
        scales: 对应的尺度 (1D array)
        wavelet: 使用的小波基
        dt: 采样时间间隔 (默认 1.0)
    
    Returns:
        重建的时域信号 (B, C, T)
    """
    print(coeffs.shape)
    B, C, S, T = coeffs.shape  # (Batch, Channels, Scales, Time)

    # 获取频率（pywt 计算的物理频率）
    frequencies = pywt.scale2frequency(wavelet, scales) / dt

    # 计算尺度权重（标准逆变换公式）
    scale_weights = np.sqrt(scales[:, None])  # (S, 1)

    # 进行加权求和（按照尺度重构）
    recon = torch.sum(
        torch.tensor(scale_weights, dtype=coeffs.dtype, device=coeffs.device) * coeffs, dim=2
    )

    return recon  # 返回重建信号

def cwt_inverse_transform(cwt_coeffs, wavelet='morl'):
    """
    计算 CWT 逆变换（注意：CWT 理论上没有完美的逆变换）
    - cwt_coeffs: 小波系数 (B, C, scales, T)
    - wavelet: 选择的小波基函数（默认 morl）
    """
    B, C, scales, T = cwt_coeffs.shape
    reconstructed_list = []

    for b in range(B):
        batch_reconstructed = []
        for c in range(C):
            coeffs = cwt_coeffs[b, c].cpu().numpy()
            scales_arr = np.arange(1, scales + 1)
            reconstructed = pywt.idwt(coeffs, scales_arr, wavelet)  # 逆变换
            batch_reconstructed.append(reconstructed)
        reconstructed_list.append(batch_reconstructed)

    return torch.tensor(reconstructed_list)  # (B, C, T)

def WADA_CWT(src_signal, trg_signal, wavelet='morl', scales=64, low=10,high=50):
    """
    连续小波变换（CWT）版本的 WADA
    - src_signal: 源信号 (B, C, T)
    - trg_signal: 目标信号 (B, C, T)
    - wavelet: 小波基（默认为 morl）
    - scales: 变换的尺度数量
    - mutate_scale_range: (low, high) 指定替换的尺度范围
    """
    src_cwt = cwt_transform(src_signal, wavelet, scales)  # (B, C, scales, T)
    trg_cwt = cwt_transform(trg_signal, wavelet, scales)

    # 替换特定尺度范围的系数
    # low, high = mutate_scale_range
    src_cwt[:, :, low:high, :] = trg_cwt[:, :, low:high, :]

    # 逆变换回时域信号
    mixed_signal = cwt_inverse_transform(src_cwt, wavelet)
    return mixed_signal.to(src_signal.dtype)

# B, C, T = 2, 342, 1800
# src = torch.randn(B, C, T)
# trg = torch.randn(B, C, T)

# fs = 1000  # 采样频率
# cutoff_freq = 7  # 截止频率(Hz)
# mixed_fs = FDA_1d_with_fs(src, trg, fs=fs, cutoff_freq=cutoff_freq)
# print(mixed_fs.shape)
# print(f"实际替换频率范围：0-{cutoff_freq}Hz (Nyquist={fs//2}Hz)")