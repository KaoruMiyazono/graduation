import os
import numpy as np
import torch 
import random
from scipy import interpolate
from tqdm import tqdm
import re
from collections import defaultdict
# from process_widar import Intel
import matplotlib.pyplot as plt
import json

def average_list(d_list):
    '''
    Args:
        d_list (list): shape [T,90]

    '''
    sum = [0.0 for _ in range(len(d_list[0]))] # for each channel
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum

def merge_timestamp(data, time_stamp, new_length = 2000):
    """align each samples time length

    Args:
        data (ndarray): input signal list with shape [T,90]
        time_stamp (list): corresponding time stamp 

    Returns:
        aligned_data(list): list whose elements have the same length
    """
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / new_length
    cur_range = time_stamp[0] + intervel
    temp_list = []
    align_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                align_data.append(average_list(temp_list))
            else:
                align_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
        temp_list.append(data[i])
    if len(temp_list) != 0:
        align_data.append(average_list(temp_list))
    if len(align_data) < new_length:
        align_data.append(data[len(time_stamp)-1])
        print("shorter than new_length, add the last element")
    return align_data[:new_length]

def resample(signal,time_stamp,newtimelen):
    """
        upsample or downsample the signal to target length
    Args:
        signal: Original signal
        time_stamp:time_stamp of the original signal
        newtimelen: the target time length
    """
    current_time_len = len(signal)
    # upsample
    if current_time_len<newtimelen:
        return upsample(signal,time_stamp,newtimelen)

    # no action
    if current_time_len == newtimelen:
        return signal

    # downsample
    if current_time_len>newtimelen:
        return merge_timestamp(signal,time_stamp,newtimelen)
    

def upsample(signal,time_stamp,newtimelen):
    '''
    根据这个位置在两个stamp之间的比例进行插值
    TODO 比较直接生成时间轴然后用np的插值函数和自己逐个插值之间时间区别
    '''
    # 需要require_new_time_stamp个元素,那么中间的间隔就需要require_new_time_stamp+1个
    require_new_time_stamp = newtimelen-len(time_stamp)
    interval = (time_stamp[-1] - time_stamp[0]) / (require_new_time_stamp+1)
    new_time_stamp = np.zeros(newtimelen)
    new_time_stamp[0] = time_stamp[0]
    #new_time_stamp[-1] = time_stamp[-1]
    current_timestamp = time_stamp[0]
    origin_time_index = 1
    new_time_index = 1
    while new_time_index<newtimelen:
        current_timestamp = current_timestamp+interval
        while current_timestamp>= time_stamp[origin_time_index] and origin_time_index<len(time_stamp)-1: 
            #一直往后面搜索，保证当当前newtimestamp内的所有oritimestamp全部被保留
            new_time_stamp[new_time_index] = time_stamp[origin_time_index]
            origin_time_index+=1  #增加oritimesamp直至大于current_timestamp
            new_time_index +=1    
        # if current_timestamp >time_stamp[-1]:#不太可能的情况是current_timestamp大于oritimestamp的最终值
        #     current_timestamp = (new_time_stamp[new_time_index-1]+time_stamp[-1])/2      
        new_time_stamp[new_time_index] = current_timestamp
        new_time_index+=1
    #print(signal.shape) # T C
    f = interpolate.interp1d(time_stamp, signal,axis=0)
    new_time_stamp = np.clip(new_time_stamp, np.min(time_stamp), np.max(time_stamp)) #我家的  可能不对
    return f(new_time_stamp) 

def rename_widar_files(folder_path):
    """
    扫描指定文件夹，将文件名从 'room_X_user_userY_ges_A_loc_B_ori_C_rx_rD.npy' 
    修改为 'room_X_user_Y_ges_A_loc_B_ori_C_rx_D.npy'。
    
    :param folder_path: 需要重命名文件的文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    files_renamed = 0
    
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        
        # 只处理 .npy 文件
        if filename.endswith(".npy"):
            new_filename = re.sub(r"user_user(\d+)", r"user_\1", filename)  # user_userX -> user_X
            new_filename = re.sub(r"rx_r(\d+)", r"rx_\1", new_filename)  # rx_rX -> rx_X
            
            new_path = os.path.join(folder_path, new_filename)
            
            # 只有当文件名发生变化时才重命名
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"✅ 重命名: {filename} -> {new_filename}")
                files_renamed += 1
    
    if files_renamed == 0:
        print("⚠️ 没有需要重命名的文件。")
    else:
        print(f"🎉 共重命名 {files_renamed} 个文件。")

def count_files_in_folder(folder_path):
    """
    统计指定文件夹内的文件数量（不包括子文件夹）。
    
    :param folder_path: 文件夹路径
    :return: 文件数量
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return 0

    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    
    print(f"📂 文件夹 {folder_path} 内共有 {file_count} 个文件。")
    return file_count


def count_files_by_user(folder_path):
    """
    统计指定文件夹下每个 user 对应的文件数量。
    
    :param folder_path: 文件夹路径
    :return: 一个字典，包含每个 user 和对应的文件数量
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return {}
    
    user_file_count = defaultdict(int)  # 用来存储每个 user 对应的文件数量
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 只统计文件，不统计子文件夹
        if os.path.isfile(file_path):
            # 假设文件名中有 'user_X' 格式，通过正则表达式提取 'user_X'
            if 'user_' in filename:
                user_match = filename.split('user_')[-1].split('_')[0]  # 提取 'user_X'
                user_file_count[user_match] += 1
    
    if user_file_count:
        print("每个 user 的文件数量：")
        for user, count in user_file_count.items():
            print(f"user_{user}: {count} 个文件")
    else:
        print("⚠️ 未找到符合条件的文件。")
    
    return user_file_count


# def count_files_by_user(folder_path, target_user='3'):
#     """
#     统计指定文件夹下每个 user 对应的文件数量，并可打印指定 user 的文件列表。
    
#     :param folder_path: 文件夹路径
#     :param target_user: 需要查看的用户（例如 'user_3'），如果为 None 则打印所有用户的文件数量
#     :return: 一个字典，包含每个 user 和对应的文件数量
#     """
#     if not os.path.exists(folder_path):
#         print(f"❌ 文件夹不存在: {folder_path}")
#         return {}
    
#     user_file_count = defaultdict(list)  # 用来存储每个 user 对应的文件列表
    
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
        
#         # 只统计文件，不统计子文件夹
#         if os.path.isfile(file_path):
#             # 假设文件名中有 'user_X' 格式，通过正则表达式提取 'user_X'
#             if 'user_' in filename:
#                 user_match = filename.split('user_')[-1].split('_')[0]  # 提取 'user_X'
#                 user_file_count[user_match].append(filename)
    
#     if target_user:
#         if target_user in user_file_count:
#             print(f"{target_user} 的文件列表：")
#             for file in user_file_count[target_user]:
#                 print(file)
#         else:
#             print(f"⚠️ 没有找到 {target_user} 的文件。")
    
#     else:
#         print("每个 user 的文件数量：")
#         for user, files in user_file_count.items():
#             print(f"user_{user}: {len(files)} 个文件")
    
#     return user_file_count


def count_files_by_conditions(folder_path, room, loc, ori, target_user=None):
    """
    统计指定文件夹下，符合指定条件（如 room, loc, ori 和 user）的文件数量。
    
    :param folder_path: 文件夹路径
    :param room: 要查询的 room，如 'room_1'
    :param loc: 要查询的 loc，如 'loc_1'
    :param ori: 要查询的 ori，如 'ori_1'
    :param target_user: 需要查看的用户（例如 'user_3'），如果为 None 则不限制 user
    :return: 符合条件的文件列表
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return []
    
    matched_files = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 只统计文件，不统计子文件夹
        if os.path.isfile(file_path):
            # 过滤文件名中包含指定条件的文件
            if room in filename and loc in filename and ori in filename:
                # 如果指定了 target_user，也要检查是否符合 user_3 格式
                if target_user and f"user_{target_user}" in filename:
                    matched_files.append(filename)
                # 如果没有指定 user，直接符合其他条件的文件也加入
                elif not target_user:
                    matched_files.append(filename)
    
    if matched_files:
        print(f"符合条件的文件数量：{len(matched_files)} 个文件")
        for file in matched_files:
            print(file)
    else:
        print(f"⚠️ 没有找到符合条件的文件。")
    
    return matched_files


def plot_csi_length_distribution(len_list, save_path="csi_length_distribution.png"):
    """
    根据 CSI 长度分布绘制直方图，并计算统计信息
    :param len_list: 存储每个样本 CSI 长度的列表
    :param save_path: 保存图片的路径
    """
    len_list = np.array(len_list)
    bins = np.arange(0, max(len_list) + 100, 100)

    mean_val = np.mean(len_list)
    max_val = np.max(len_list)
    min_val = np.min(len_list)
    median_val = np.median(len_list)

    plt.figure(figsize=(10, 5))
    plt.hist(len_list, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("CSI Length (每 100 一个区间)")
    plt.ylabel("样本数量")
    plt.title("CSI 长度分布")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    text_str = f"均值: {mean_val:.2f}\n最大值: {max_val}\n最小值: {min_val}\n中位数: {median_val}"
    plt.text(0.7 * max(len_list), max(plt.hist(len_list, bins=bins)[0]) * 0.7, text_str, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))

    # **保存图片**
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存为 {save_path}")

    plt.show()

    return {"mean": mean_val, "max": max_val, "min": min_val, "median": median_val}


# def get_widar_length_distribution(file_path):
#      len_list=[]
#      cnt=0
#      with open(file_path, 'r') as f:
#         for line in f:
#             line = line.rstrip()
#             # print(line)
            
#             widar_data_single=Intel(file=line,nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
#             widar_data_single.read()
#             csi=widar_data_single.get_scaled_csi()
#             len_list.append(csi.shape[0])
#             cnt=cnt+1
#             print(cnt)
#         return len_list





