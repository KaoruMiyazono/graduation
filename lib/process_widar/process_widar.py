
import os 
import numpy as np
import json 
from utils import resample,rename_widar_files,count_files_in_folder,count_files_by_user,count_files_by_user,count_files_by_conditions
import matplotlib.pyplot as plt


gesture_set=['Push&Pull', 'Sweep', 'Clap', 'Slide', 'Draw-O(Horizontal)', 'Draw-Zigzag(Horizontal)']
def read_json_csi(file):
#输入文件名字 输出json文件的数据结构形式
    with open(file, 'r') as file:
        data = json.load(file)
        # print(data['20181109']['room'])
    return data

def get_new_json(data):
    filtered_data = {}
    for date, details in data.items():
        room = details["room"]
        users = details["users"]
        
        # 筛选每个用户的手势映射
        filtered_users = {}
        for user, user_data in users.items():
            gesture_map = user_data["gesture_map"]
            
            # 筛选出 gesture_map 中属于 gesture_set 的手势
            filtered_gesture_map = {
                gesture: gesture_id
                for gesture, gesture_id in gesture_map.items()
                if gesture in gesture_set
            }
            
            # 如果筛选后的 gesture_map 不为空，则保留该用户
            if filtered_gesture_map:
                filtered_users[user] = {
                    "gesture_map": filtered_gesture_map
                }
        
        # 如果筛选后的 users 不为空，则保留该日期
        if filtered_users:
            filtered_data[date] = {
                "room": room,
                "users": filtered_users
            }
    
    # 将筛选后的数据写入新的 JSON 文件
    with open('filtered_csi_summary.json', 'w') as file:
        json.dump(filtered_data, file, indent=4)
    
    print("筛选完成，结果已写入 filtered_csi_summary.json")

class Intel:
    """
    class used to read csi data from .dat file
    This implementation is modified from
    https://github.com/citysu/csiread/blob/master/examples/csireadIntel5300.py
    """
    def __init__(self, file, nrxnum=3, ntxnum=2, pl_len=0, if_report=True):
        self.file = file
        self.nrxnum = nrxnum
        self.ntxnum = ntxnum
        self.pl_len = pl_len    # useless
        self.if_report = if_report #useless 
        # print(file)
        if not os.path.isfile(file):
            raise Exception("error: file does not exist, Stop!\n")

    def __getitem__(self, index):
        """Return contents of 0xbb packets"""
        ret = {
            "timestamp_low": self.timestamp_low[index],
            "bfee_count": self.bfee_count[index],
            "Nrx": self.Nrx[index],
            "Ntx": self.Ntx[index],
            "rssi_a": self.rssi_a[index],
            "rssi_b": self.rssi_b[index],
            "rssi_c": self.rssi_c[index],
            "noise": self.noise[index],
            "agc": self.agc[index],
            "perm": self.perm[index],
            "rate": self.rate[index],
            "csi": self.csi[index]
        }
        return ret

    def read(self):
        f = open(self.file, 'rb')
        if f is None:
            f.close()
            return -1

        lens = os.path.getsize(self.file)
        btype = np.int_
        #self.timestamp_low = np.zeros([lens//95], dtype = btype)
        self.timestamp_low = np.zeros([lens//95], dtype = np.int64)
        self.bfee_count = np.zeros([lens//95], dtype = btype)
        self.Nrx = np.zeros([lens//95], dtype = btype)
        self.Ntx = np.zeros([lens//95], dtype = btype)
        self.rssi_a = np.zeros([lens//95], dtype = btype)
        self.rssi_b = np.zeros([lens//95], dtype = btype)
        self.rssi_c = np.zeros([lens//95], dtype = btype)
        self.noise = np.zeros([lens//95], dtype = btype)
        self.agc = np.zeros([lens//95], dtype = btype)
        self.perm = np.zeros([lens//95, 3], dtype = btype)
        self.rate = np.zeros([lens//95], dtype = btype)
        self.csi = np.zeros([lens//95, 30, self.nrxnum, self.ntxnum], dtype = np.complex_)

        cur = 0
        count = 0
        while cur < (lens-3):
            temp = f.read(3)
            field_len = temp[1]+(temp[0]<<8)
            code = temp[2]
            cur += 3
            if code == 187:
                buf = f.read(field_len - 1)
                if len(buf) != field_len - 1:
                    break
                self.timestamp_low[count] = int.from_bytes(buf[:4], 'little')
                self.bfee_count[count] = int.from_bytes(buf[4:6], 'little')
                assert self.nrxnum == buf [8] # check the pre given nrx number is correct
                assert self.ntxnum == buf [9] # check the pre given ntx number is correct
                self.Nrx[count] = buf[8]
                self.Ntx[count] = buf[9]
                self.rssi_a[count] = buf[10]
                self.rssi_b[count] = buf[11]
                self.rssi_c[count] = buf[12]
                self.noise[count] = int.from_bytes(buf[13:14], 'little', signed=True)
                self.agc[count] = buf[14]
                self.rate[count] = int.from_bytes(buf[18:20], 'little')

                self.perm[count, 0] = buf[15] & 0x3
                self.perm[count, 1] = (buf[15] >> 2) & 0x3
                self.perm[count, 2] = (buf[15] >> 4) & 0x3

                index = 0
                payload = buf[20:]
                for i in range(30):
                    index += 3
                    remainder = index & 0x7
                    for j in range(buf[8]):
                        for k in range(buf[9]):
                            a = (payload[index // 8] >> remainder) | (payload[index // 8 + 1] << (8 - remainder)) & 0xff
                            b = (payload[index // 8 + 1] >> remainder) | (payload[index // 8 + 2] << (8 - remainder)) & 0xff
                            if a >= 128:
                                a -= 256
                            if b >= 128:
                                b -= 256
                            self.csi[count, i, self.perm[count, j], k] = a + b * 1.j
                            index += 16
                count += 1
            else:
                f.seek(field_len - 1, os.SEEK_CUR)
            cur += field_len - 1
        f.close()
        self.timestamp_low = self.timestamp_low[:count]
        self.bfee_count = self.bfee_count[:count]
        self.Nrx = self.Nrx[:count]
        self.Ntx = self.Ntx[:count]
        self.rssi_a = self.rssi_a[:count]
        self.rssi_b = self.rssi_b[:count]
        self.rssi_c = self.rssi_c[:count]
        self.noise = self.noise[:count]
        self.agc = self.agc[:count]
        self.perm = self.perm[:count, :]
        self.rate = self.rate[:count]
        self.csi = self.csi[:count, :, :, :]
        self.count = count

    def get_total_rss(self):
        """Calculates the Received Signal Strength (RSS) in dBm from CSI"""
        rssi_mag = np.zeros_like(self.rssi_a, dtype=float)
        rssi_mag += self.__dbinvs(self.rssi_a)
        rssi_mag += self.__dbinvs(self.rssi_b)
        rssi_mag += self.__dbinvs(self.rssi_c)
        ret = self.__db(rssi_mag) - 44 - self.agc
        return ret

    def get_scaled_csi(self):
        """Converts CSI to channel matrix H"""
        csi = self.csi
        csi_sq = (csi * csi.conj()).real
        csi_pwr = np.sum(csi_sq, axis=(1, 2, 3))
        rssi_pwr = self.__dbinv(self.get_total_rss())

        scale = rssi_pwr / (csi_pwr / 30)

        noise_db = self.noise
        thermal_noise_pwr = self.__dbinv(noise_db)
        thermal_noise_pwr[noise_db == -127] = self.__dbinv(-92)

        quant_error_pwr = scale * (self.Nrx * self.Ntx)
        total_noise_pwr = thermal_noise_pwr + quant_error_pwr

        ret = self.csi * np.sqrt(scale / total_noise_pwr).reshape(-1, 1, 1, 1)
        ret[self.Ntx == 2] *= np.sqrt(2)
        ret[self.Ntx == 3] *= np.sqrt(self.__dbinv(4.5))
        ret = ret.conj()
        return ret

    def get_scaled_csi_sm(self):
        """Converts CSI to channel matrix H
        This version undoes Intel's spatial mapping to return the pure
        MIMO channel matrix H.
        """
        ret = self.get_scaled_csi()
        ret = self.__remove_sm(ret)
        return ret

    def __dbinvs(self, x):
        """Convert from decibels specially"""
        ret = np.power(10, x / 10)
        ret[ret == 1] = 0
        return ret

    def __dbinv(self, x):
        """Convert from decibels"""
        ret = np.power(10, x / 10)
        return ret

    def __db(self, x):
        """Calculates decibels"""
        ret = 10 * np.log10(x)
        return ret

    def __remove_sm(self, scaled_csi):
        """Actually undo the input spatial mapping"""
        sm_1 = 1
        sm_2_20 = np.array([[1, 1],
                            [1, -1]]) / np.sqrt(2)
        sm_2_40 = np.array([[1, 1j],
                            [1j, 1]]) / np.sqrt(2)
        sm_3_20 = np.array([[-2 * np.pi / 16, -2 * np.pi / (80 / 33), 2 * np.pi / (80 / 3)],
                            [ 2 * np.pi / (80 / 23), 2 * np.pi / (48 / 13), 2 * np.pi / (240 / 13)],
                            [-2 * np.pi / (80 / 13), 2 * np.pi / (240 / 37), 2 * np.pi / (48 / 13)]])
        sm_3_20 = np.exp(1j * sm_3_20) / np.sqrt(3)
        sm_3_40 = np.array([[-2 * np.pi / 16, -2 * np.pi / (80 / 13), 2 * np.pi / (80 / 23)],
                            [-2 * np.pi / (80 / 37), -2 * np.pi / (48 / 11), -2 * np.pi / (240 / 107)],
                            [ 2 * np.pi / (80 / 7), -2 * np.pi / (240 / 83), -2 * np.pi / (48 / 11)]])
        sm_3_40 = np.exp(1j * sm_3_40) / np.sqrt(3)
    
        ret = scaled_csi
        for i in range(self.count):
            M = self.Ntx[i]
            if (int(self.rate[i]) & 2048) == 2048:
                if M == 3:
                    sm = sm_3_40
                elif M == 2:
                    sm = sm_2_40
                else:
                    sm = sm_1
            else:
                if M == 3:
                    sm = sm_3_20
                elif M == 2:
                    sm = sm_2_20
                else:
                    sm = sm_1
            if sm != 1:
                ret[i, :, :, :M] = ret[i, :, :, :M].dot(sm.T.conj())
        return ret
    
def find_r2_dat_files(root_dir, output_file, json_file):
    # 加载 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 打开输出文件
    with open(output_file, 'w') as f:
        # 遍历根目录下的所有文件夹
        for folder in sorted(os.listdir(root_dir)):
            if not folder.startswith("2018"):
                continue  # 仅处理2018开头的文件夹
            
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # 获取该日期对应的手势映射信息
            if folder in data:
                gesture_info = data[folder]
            else:
                continue  # 如果日期不在 JSON 中，跳过
            
            # 遍历子目录和文件
            for subdir, sss, files in os.walk(folder_path):
                # 提取用户 ID（假设子目录名为用户 ID）
                # print(sss)
                # print(subdir)
                user_id = os.path.basename(subdir)
                # print(user_id)
                # exit(0)
                
                # 获取该用户的手势映射
                if "users" in gesture_info and user_id in gesture_info["users"]:
                    gesture_map = gesture_info["users"][user_id]["gesture_map"]
                    # print(gesture_map)
                else:
                    continue  # 如果用户不在手势映射中，跳过
                
                # 遍历文件
                for file in files:
                    # print(file)
                    if file.endswith("r6.dat"):  # 仅处理 r2.dat 文件
                        parts = file.split('-')
                        if len(parts) < 6:
                            continue  # 确保文件名格式正确
                        
                        gesture_type = parts[1]  # 提取 gesture type

                        gesture_type = int(gesture_type)
                        # 检查手势类型是否在 gesture_map 中
                        if gesture_type in gesture_map.values():
                            file_path = os.path.join(subdir, file)
                            print(file_path)
                            f.write(file_path + "\n")



def count_samples(file_path):
    user_counts = {"user1": 0, "user2": 0, "user3": 0, "user4": 0, "user5": 0, "user6": 0,"user7": 0, "user8": 0, "user9": 0,"user10": 0,"user11": 0,"user12": 0,"user13": 0,"user14": 0,"user15": 0,"user16": 0,"user17": 0,}
    with open(file_path, 'r') as f:
        for line in f:
            user_folder = line.strip().split("/")[-2]  # 获取 user 目录名
            if user_folder in user_counts:
                user_counts[user_folder] += 1
    return user_counts


def count_locations_orientations(file_path):
    #计算 有多少个 location和 朝向
    users_of_interest = {"user6", "user7", "user8", "user9"}
    location_counts = {user: set() for user in users_of_interest}
    orientation_counts = {user: set() for user in users_of_interest}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = os.path.basename(line).split('-')
            if len(parts) < 6:
                continue  # 确保文件名格式正确
            
            user_id = parts[0]  # 提取用户ID
            if user_id not in users_of_interest:
                continue
            
            location = int(parts[2])  # 提取location
            orientation = int(parts[3])  # 提取orientation
            
            location_counts[user_id].add(location)
            orientation_counts[user_id].add(orientation)
    
    # 输出统计结果
    with open("location_orientation_stats.txt", 'w') as stats_file:
        for user in users_of_interest:
            stats_file.write(f"{user}: {len(location_counts[user])} locations, {len(orientation_counts[user])} orientations\n")



def extract_sample_info(file_path, json_file):
    """
    从文件路径和 JSON 文件中提取样本信息，并生成格式化字符串。
    
    参数:
    - file_path: 文件路径（如 `/home/zhengzhiyong/WiSR-main/WIDAR/20181109/user3/user3-4-3-2-4-r2.dat`）。
    - json_file: JSON 文件路径，包含房间信息。
    
    返回:
    - 格式化字符串（如 `room_1_user_user3_ges_4_loc_3_ori_2_rx_r2`）。
    """
    # 从文件路径中提取文件名
    file_name = os.path.basename(file_path)
    
    # 提取用户 ID、手势类型、躯干位置、面部朝向、重复次数和接收器 ID
    parts = file_name.split('-')
    if len(parts) < 6:
        raise ValueError("文件名格式不正确")
    
    user = parts[0]  # 用户 ID（如 user3）
    ges = parts[1]   # 手势类型（如 4）
    loc = parts[2]   # 躯干位置（如 3）
    ori = parts[3]   # 面部朝向（如 2）
    repetation=parts[4]
    rx = parts[5].split('.')[0]  # 接收器 ID（如 r2）
    
    # 从 JSON 文件中提取 room
    date = file_path.split('/')[-3]  # 提取日期（如 20181109）
    # print(date)
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if date in data:
        room = data[date]["room"]
        gesture_map_info=data[date]["users"][user]["gesture_map"]

        gestre_string = next((k for k, v in gesture_map_info.items() if v == int(ges)), None)
        num_user= int(user[4:])

    else:
        raise ValueError(f"日期 {date} 不在 JSON 文件中")
    
    # 生成格式化字符串
    formatted_str = f"room_{room}_user_{num_user}_ges_{gestre_string}_loc_{loc}_ori_{ori}_rx_{rx}_re_{repetation}"
    return formatted_str

def get_csi_and_save_with_numpy(file_path, json_file):
    string_list = []
    error_file = []
    file_map = {}  # 存储 .npy 文件名 -> .dat 文件路径
    cnt = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()

            widar_data_single = Intel(file=line, nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
            widar_data_single.read()
            csi = widar_data_single.get_scaled_csi()

            # amp = np.abs(csi[:, :, 0:3, :])  # shape [T,30,3,1]
            # print(amp.shape)
            # phase = np.angle(csi[:, :, 0:3, :])  # shape [T,30,3,1]
            ref_antenna = np.expand_dims(np.conjugate(csi[:,:,2,:]),axis=2)
            res_antenna = csi[:,:,0:2,:]  # when training discarding the reference antenna 1
            csi_ratio = res_antenna*ref_antenna #shape [T,30,3,1] complex number
            csi_ratio_amp = np.abs(csi_ratio)  #shape [T,30,3,1]
            csi_ratio_pha = np.angle(csi_ratio)  #shape [T,30,3,1]
            record = np.concatenate((csi_ratio_amp,csi_ratio_pha),axis=-1) #shape [T,30,3,4]
            print(record.shape)
            print(record[:,:,:,0]==csi_ratio_amp[:,:,:,0])
            print(record[:,:,:,1]==csi_ratio_pha[:,:,:,0])
            # exit(0)
            # record = np.concatenate((amp, phase), axis=-1)  # shape [T,30,3,2]
            # print(record[:,:,:,0]==amp[:,:,:,0])
            # exit(0)

            time_stamp = widar_data_single.timestamp_low
            record = resample(record, time_stamp, 2500)
            record = np.array(record)
            # print(record.shape)
            # exit(0)

            ans = extract_sample_info(line, json_file)
            save_dir = "/home/zhengzhiyong/WiSR-main/data/Widar3/r6_conj"
            file_name = ans + ".npy"  # 目标 .npy 文件名
            save_path = os.path.join(save_dir, file_name)

            if os.path.exists(save_path):
                error_file.append(line)  # 记录重复文件的 .dat 文件路径
                print(f"有重复: {file_name} 已存在")

                # 记录当前 .npy 文件对应的 .dat 文件路径
                if file_name in file_map:
                    old_dat_file = file_map[file_name]
                    print(f"文件 {file_name} 之前对应的 .dat 文件是: {old_dat_file}")
                    print(f"文件 {file_name} 现在的 .dat 文件是: {line}")

                    # 追加写入详细错误日志
                    with open("error.txt", "a") as f:
                        f.write(f"文件 {file_name} 重复:\n")
                        f.write(f"  - 之前的 .dat 文件: {old_dat_file}\n")
                        f.write(f"  - 现在的 .dat 文件: {line}\n\n")
                else:
                    file_map[file_name] = line  # 记录新的 .dat 文件路径
                    with open("error.txt", "a") as f:
                        f.write(f"文件 {file_name} 重复: {line}\n")

            else:
                os.makedirs(save_dir, exist_ok=True)
                np.save(save_path, record)  # 保存数据
                file_map[file_name] = line  # 记录当前 .dat 文件路径
            
            cnt += 1
            print(f"数组已保存到: {save_path}")
            print(f"数组大小为：{record.shape}")
            print(f"已经处理了 {cnt}")

            string_list.append(ans)

    print(f"总共处理了 {len(string_list)} 个文件")
    print(f"发现 {len(error_file)} 个重复文件")
    








if __name__ == "__main__":
    root_directory = "/home/zhengzhiyong/WiSR-main/WIDAR"
    filter_json="/home/zhengzhiyong/WiSR-main/graduation/lib/process_widar/filtered_csi_summary.json"
    # output_dir="/home/zhengzhiyong/WiSR-main/graduation/lib/process_widar/r2_files.txt"
    output_dir="/home/zhengzhiyong/WiSR-main/graduation/lib/process_widar/r6_files.txt"
    r2_files="/home/zhengzhiyong/WiSR-main/graduation/lib/process_widar/r2_files.txt"
    r6_files="/home/zhengzhiyong/WiSR-main/graduation/lib/process_widar/r6_files.txt"
    npy_r6_no_conj="/home/zhengzhiyong/WiSR-main/data/widar/r6_noconj"
    npy_r6_conj="/home/zhengzhiyong/WiSR-main/data/Widar3/r6_conj"
    npy_r6="/home/zhengzhiyong/WiSR-main/data/widar/r6"
    # output_txt = "r2_files.txt"
    # find_r2_dat_files(root_directory, output_txt)
    # output_txt = "/home/zhengzhiyong/WiSR-main/graduation/lib/r2_files.txt"
    # 
    # # 2 3 5 7 8 9
    # for user, count in user_sample_counts.items():
    #     print(f"{user}: {count} samples")
    # print(f"File paths saved to {output_txt}")

    # 
    # widar_data_single=Intel(file="/home/zhengzhiyong/WiSR-main/WIDAR/20181128/user6/user6-3-2-2-1-r2.dat",nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
    # widar_data_single.read()
    # print(csi)
    # csi_data=widar_data_single.get_scaled_csi()
    # print(csi_data)
    # print(type(csi_data))
    # print(csi_data.shape)
    # print(type(csi))

    # data=read_json_csi("widar_process.json")
    # get_new_json(data)
    # find_r2_dat_files(root_directory,output_dir,filter_json)
    # count_locations_orientations(r2_files)
    # user_sample_counts = count_samples(r6_files)
    # print(user_sample_counts)
    get_csi_and_save_with_numpy(r6_files,filter_json)
    # len_list = get_widar_length_distribution(r2_files)
    # stats = plot_csi_length_distribution(len_list, save_path="csi_length_hist.png")

    # print("统计信息:", stats)
    # rename_widar_files(npy_r6)
    # count_files_in_folder(npy_r6_no_conj)
    # count_files_by_user(npy_r6_no_conj)
    # count_files_by_user(npy_r6)
    # count_files_by_conditions(npy_r6_conj,'room_3','loc_4','ori_2','3')

    
