
import torch
def high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=1000, cutoff_freq=10):
    """ 使用物理频率控制低频区域
    Args:
        amp_src: 源幅度谱 (B, C, L)
        amp_trg: 目标幅度谱 
        fs: 采样频率 (Hz)
        cutoff_freq: 截止频率 (Hz)
    """
    B, C, t = amp_src.shape
    nyquist = fs / 2  # Nyquist频率
    total_freq_bins = t  # FFT后的频点总数（rfft的特殊性需处理）
    # print(B,C,t)
    # 计算截止频率对应的频点位置
    cutoff_bin = int( (cutoff_freq / nyquist) * t )  # 只考虑单侧频谱
    # print(cutoff_bin)
    
    # 替换高频区域（考虑rfft的对称性）
    amp_src[..., cutoff_bin:] = amp_trg[..., cutoff_bin:]
    return amp_src

def FDA_1d_with_fs(src_signal, trg_signal, fs=1000, cutoff_freq=70):
    # 输入形状: (B, C, T)
    src = src_signal
    trg = trg_signal

    # 计算RFFT（实数信号FFT）
    fft_src = torch.fft.rfft(src, dim=-1, norm='forward')  # 输出形状 (B, C, L+1)
    fft_trg = torch.fft.rfft(trg, dim=-1, norm='forward')
    # rfreqs = torch.fft.rfftfreq(1800, 1/1000)  # 得到非负频率
    # print(src.shape)
    # print(rfreqs)
    
    # 分解幅度和相位
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg,pha_trg = torch.abs(fft_trg),torch.angle(fft_trg)
    
    # 使用物理频率进行低频替换
    # amp_src_mutated = high_freq_mutate_1d_with_fs(amp_src, amp_trg, fs=fs, cutoff_freq=cutoff_freq)
    pha_src_mutated = high_freq_mutate_1d_with_fs(pha_src, pha_trg, fs=fs, cutoff_freq=cutoff_freq)
    
    # 重建信号
    fft_mixed = torch.polar(amp_src, pha_src_mutated)
    mixed = torch.fft.irfft(fft_mixed, n=src.size(-1), dim=-1, norm='forward')
    
    return mixed.to(src_signal.dtype)


# B, C, T = 2, 342, 1800
# src = torch.randn(B, C, T)
# trg = torch.randn(B, C, T)

# fs = 1000  # 采样频率
# cutoff_freq = 7  # 截止频率(Hz)
# mixed_fs = FDA_1d_with_fs(src, trg, fs=fs, cutoff_freq=cutoff_freq)
# print(mixed_fs.shape)
# print(f"实际替换频率范围：0-{cutoff_freq}Hz (Nyquist={fs//2}Hz)")