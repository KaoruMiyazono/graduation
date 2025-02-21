
import numpy as np
import re,os,pickle
import scipy.io as scio
import zarr
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import stft
def plot_amplitude_over_time(amplitude_data, antenna, subcarrier, save_path="amplitude_plot.png"):
    """
    绘制固定天线和子载波，振幅随时间变化的图，并将图保存到当前目录。

    参数:
    amplitude_data: 振幅数据 (shape: 1800, 3, 114)
    antenna: 选择的天线索引（0、1、2）
    subcarrier: 选择的子载波索引（0 到 113）
    save_path: 保存图像的路径（默认保存为当前目录中的 amplitude_plot.png）
    """
    # 提取指定天线和子载波的振幅数据
    time = np.arange(1800)  # 时间轴（单位：ms）
    amplitude = amplitude_data[:, antenna, subcarrier]  # 获取固定天线和子载波的振幅

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(time, amplitude, label=f"Antenna {antenna+1}, Subcarrier {subcarrier+1}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title(f'Amplitude vs Time for Antenna {antenna+1} and Subcarrier {subcarrier+1}')
    plt.grid(True)
    plt.legend()

    # 保存图像到当前目录
    plt.savefig(save_path)
    plt.close()  # 关闭图像，释放内存
def read_csi_sample(index):
    root_dir_amp='/home/zhengzhiyong/Wifi/dataset/CSIDA/dataset/csi_data_amp'
    root_dir_pha='/home/zhengzhiyong/Wifi/dataset/CSIDA/dataset/csi_data_pha'
    csi_amp = zarr.open(Path(root_dir_amp).as_posix(), mode="r")
    csi_pha = zarr.open(Path(root_dir_pha).as_posix(), mode="r")
    amp=csi_amp[index]
    pha=csi_pha[index]
    # Z=csi_raw[index]
    # amplitude = np.abs(Z)
    # pha=np.angle(Z)
    return amp,pha,pha



def compute_stft(amplitude_data, nperseg=256):
    """
    对振幅数据 (shape: 1800, 3, 114) 计算短时傅里叶变换 (STFT)，返回时频表示。
    
    参数:
    amplitude_data: 振幅数据，形状 (1800, 3, 114)
    nperseg: 每个时间段的样本数（窗口大小）
    
    返回:
    stft_data: 计算后的 STFT 数据，形状 (频率数, 时间步数, 天线数, 子载波数)
    """
    # 获取振幅数据的形状
    time_steps, num_antennas, num_subcarriers = amplitude_data.shape
    
    # 确定 STFT 的时间步数
    time_steps_stft = 16  # 计算时间步数
    
    # 计算 STFT
    stft_data = np.zeros((nperseg//2 + 1, time_steps_stft, num_antennas, num_subcarriers), dtype=complex)
    
    for antenna in range(num_antennas):
        for subcarrier in range(num_subcarriers):
            # 对每个天线和子载波的振幅数据进行 STFT
            f, t, Zxx = stft(amplitude_data[:, antenna, subcarrier], nperseg=nperseg,fs=1000)
            # print(Zxx.shape)
            
            # 填充 STFT 数据
            stft_data[:, :, antenna, subcarrier] = Zxx[:, :time_steps_stft]
            
    return stft_data,f,t



def plot_spectrogram(stft_data, antenna, subcarrier, fs, nperseg, noverlap=None, 
                     freq_lim=None, cmap='viridis', save_path="spectrogram.png",f=None,t=None):
    """
    绘制给定天线和子载波的时频图（物理正确的坐标轴刻度）.
    
    参数:
    stft_data : numpy.ndarray
        STFT 数据，形状 (频率数, 时间步数, 天线数, 子载波数)
    antenna : int
        天线索引 (0-based)
    subcarrier : int
        子载波索引 (0-based)
    fs : float
        采样频率 (Hz)
    nperseg : int
        STFT 窗口长度（样本数）
    noverlap : int, optional
        窗口重叠样本数，默认 nperseg//8
    freq_lim : tuple, optional
        频率轴显示范围 (low, high)，默认全范围
    cmap : str, optional
        颜色映射，默认 'viridis'
    save_path : str, optional
        图像保存路径，默认 "spectrogram.png"
    """
    # 参数校验
    assert 0 <= antenna < stft_data.shape[2], f"天线索引 {antenna} 超出范围 [0, {stft_data.shape[2]-1}]"
    assert 0 <= subcarrier < stft_data.shape[3], f"子载波索引 {subcarrier} 超出范围 [0, {stft_data.shape[3]-1}]"
    # 提取幅度数据
    amplitude = np.abs(stft_data[:, :, antenna, subcarrier])
    # 计算频率轴 (根据 scipy.signal.stft 规则)
    # frequency = np.linspace(0, fs/2, stft_data.shape[0])
    # 计算时间轴 (窗口中心时间点)
    # noverlap = noverlap if noverlap is not None else nperseg // 2
    # step = nperseg - noverlap  # 窗口移动步长
    # time = (np.arange(amplitude.shape[1]) * step + nperseg//2) / fs  # 窗口中心时间
    # 绘图
    plt.figure(figsize=(12, 6))
    # plt.pcolormesh(time, frequency, amplitude, shading='gouraud', cmap=cmap)
    plt.pcolormesh(t, f, amplitude, shading='gouraud', cmap=cmap)
    # 坐标轴标签和标题
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Frequency [Hz]', fontsize=12)
    plt.title(f'Antenna {antenna+1}, Subcarrier {subcarrier+1}\n'
              f'Window: {nperseg} samples, Overlap: {noverlap} samples', fontsize=14)
    # 频率轴范围设置
    if freq_lim is not None:
        plt.ylim(0,freq_lim)
    else:
        plt.ylim(0, fs/2)  # 默认显示全范围
    # 颜色条
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', rotation=270, labelpad=15)
    # 优化布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_selected_spectrograms(stft_data, fs, nperseg, antennas, subcarriers, noverlap=None, 
                               freq_lim=None, cmap='viridis', save_path="selected_spectrograms.png", f=None, t=None):
    """
    绘制指定天线和子载波的时频图，并显示在一张图上。

    参数:
    stft_data : numpy.ndarray
        STFT 数据，形状 (频率数, 时间步数, 天线数, 子载波数)
    fs : float
        采样频率 (Hz)
    nperseg : int
        STFT 窗口长度（样本数）
    antennas : list of int
        要绘制的天线索引列表（0-based）
    subcarriers : list of int
        要绘制的子载波索引列表（0-based）
    noverlap : int, optional
        窗口重叠样本数，默认 nperseg//2
    freq_lim : tuple, optional
        频率轴显示范围 (low, high)，默认全范围
    cmap : str, optional
        颜色映射，默认 'viridis'
    save_path : str, optional
        图像保存路径，默认 "selected_spectrograms.png"
    f : numpy.ndarray, optional
        频率轴数据，如果为 None 则自动计算
    t : numpy.ndarray, optional
        时间轴数据，如果为 None 则自动计算
    """
    # 参数校验
    num_antennas = stft_data.shape[2]  # 天线数量
    num_subcarriers = stft_data.shape[3]  # 子载波数量
    assert all(0 <= ant < num_antennas for ant in antennas), "天线索引超出范围"
    assert all(0 <= sub < num_subcarriers for sub in subcarriers), "子载波索引超出范围"

    # 设置子图布局
    num_rows = len(antennas)
    num_cols = len(subcarriers)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), squeeze=False)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # 遍历指定的天线和子载波
    for i, antenna in enumerate(antennas):
        for j, subcarrier in enumerate(subcarriers):
            # 提取幅度数据
            amplitude = np.abs(stft_data[:, :, antenna, subcarrier])

            # 绘制时频图
            ax = axes[i, j]
            c = ax.pcolormesh(t, f, amplitude, shading='gouraud', cmap=cmap)

            # 设置标题和坐标轴
            ax.set_title(f'Ant {antenna+1}, Sub {subcarrier+1}', fontsize=10)
            ax.set_xlabel('Time [s]', fontsize=8)
            ax.set_ylabel('Freq [Hz]', fontsize=8)

            # 设置频率轴范围
            if freq_lim is not None:
                ax.set_ylim(0, freq_lim)
            else:
                ax.set_ylim(0, fs/2)

            # 添加颜色条
            fig.colorbar(c, ax=ax)

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

amp,pha,Z=read_csi_sample(400)
# plot_amplitude_over_time(amp,0,0)
stft_data,f,t=compute_stft(pha)
print(stft_data.shape)
# print(stft_data[:,:,1,100])
# print(f)
# print(t)
antenna, subcarrier=0,20
# plot_spectrogram(stft_data,1,100,1000,256,128,freq_lim=20,f=f,t=t)
antennas_to_plot = [0, 1, 2]  # 绘制天线 1、2、3
subcarriers_to_plot = [0, 20, 40, 60,80,100]  # 绘制子载波 1、11、21、31
plot_selected_spectrograms(
    stft_data=stft_data,
    fs=1000,            # 采样率 1000 Hz
    nperseg=256,        # 窗口长度 256
    antennas=antennas_to_plot,  # 指定天线
    subcarriers=subcarriers_to_plot,  # 指定子载波
    noverlap=128,       # 重叠 128 样本
    freq_lim=400,       # 频率轴显示范围 0~500 Hz
    cmap='jet',         # 颜色映射
    save_path="selected_spectrograms.png",  # 保存路径
    f=f,                # 频率轴数据
    t=t                 # 时间轴数据
)