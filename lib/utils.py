import scipy.io as scio
import numpy as np 
import chardet


def write_result_to_txt(args,results):
    file_name=args.output_dir+args.results_file
    with open(file_name,"a") as file:
        file.write("output_dir:"+args.output_dir+"\n")
        file.write("source_domains:"+str(args.source_domains)+"\n")
        file.write("target_domains:"+str(args.target_domains)+"\n")
        file.write("n_train:"+str(results[0])+"\n")
        file.write("n_test:"+str(results[1])+"\n")
        file.write("best_acc:"+str(results[2])+"\n")
        file.write("\n")


def read_widar_test(file_name):
    with open('/home/zhengzhiyong/WiSR-main/WIDAR/20181109/user1/user1-1-1-3-5-r1.dat', 'rb') as file:
        content = file.read()
        print(content)  # 打印原始字节数据


if __name__ == "__main__":
    file_name="/home/zhengzhiyong/WiSR-main/WIDAR/20181109/user1/user1-1-1-3-5-r1.dat"
    read_widar_test(file_name=file_name)