
import os 
import numpy as np

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
    
def find_r2_dat_files(root_dir, output_file):
    # 定义要提取的手势信息
    gesture_dict = {
        "20181109": [1,2,3,4],
        "20181115": [1,2,3],
        "20181117": [1,2,3],
        "20181118": [1,2,3],
        "20181121": [1,2,3],
        "20181127": [1,2,3],
        "20181128": [1,2,3,4,5],
        "20181130": [1,2,3,4,5,6],
        "20181204": [1,2,3,4,5,6],
        "20181205": {"user2": [1, 2], "user3": [1,2,3]},
        "20181208": "all",
        "20181209": "all",
        "20181211": "all"
    }
    
    with open(output_file, 'w') as f:
        for folder in sorted(os.listdir(root_dir)):
            if not folder.startswith("2018"):
                continue  # 仅处理2018开头的文件夹
            
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            gesture_filter = None
            if folder in gesture_dict:
                gesture_filter = gesture_dict[folder]
            else:
                continue
            
            for subdir, _, files in os.walk(folder_path):
                last_part = os.path.basename(subdir)  # 获取当前子目录名
                user_id = last_part.split('-')[0]  # 提取 user ID
                # print(user_id)
                
                # 获取该日期对应的手势列表
                if isinstance(gesture_filter, dict):  # 针对特定用户
                    if user_id in gesture_filter:
                        allowed_gestures = gesture_filter[user_id]
                    else:
                        continue
                elif gesture_filter == "all":
                    allowed_gestures = "all"
                else:
                    allowed_gestures = gesture_filter
                
                for file in files:
                    if file.endswith("r2.dat"):
                        parts = file.split('-')
                        if len(parts) < 6:
                            continue  # 确保文件名格式正确
                        
                        gesture_type = int(parts[1])  # 提取gesture type
                        
                        # 过滤符合条件的手势类型
                        if allowed_gestures == "all" or gesture_type in allowed_gestures:
                            file_path = os.path.join(subdir, file)
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



if __name__ == "__main__":
    # root_directory = "/home/zhengzhiyong/WiSR-main/WIDAR"
    # output_txt = "r2_files.txt"
    # find_r2_dat_files(root_directory, output_txt)
    # output_txt = "/home/zhengzhiyong/WiSR-main/graduation/lib/r2_files.txt"
    # user_sample_counts = count_samples(output_txt)
    # # 2 3 5 7 8 9
    # for user, count in user_sample_counts.items():
    #     print(f"{user}: {count} samples")
    # print(f"File paths saved to {output_txt}")

    # count_locations_orientations("/home/zhengzhiyong/WiSR-main/graduation/lib/r2_files.txt")
    widar_data_single=Intel(file="/home/zhengzhiyong/WiSR-main/WIDAR/20181128/user6/user6-3-2-2-1-r2.dat",nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
    widar_data_single.read()
    # print(csi)
    csi_data=widar_data_single.get_scaled_csi()
    print(csi_data)
    print(type(csi_data))
    print(csi_data.shape)
    # print(type(csi))