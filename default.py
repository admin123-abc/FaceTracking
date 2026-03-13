import serial
import time

def reset_servos():
    # 设定目标硬件归中参数
    # 基于HTD85H规格：0-1000对应0-240度，500为绝对物理中点
    center_signal = 500
    
    try:
        # 初始化下位机通讯链路
        # 波特率需与下位机固件保持115200一致
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        
        # 预留总线初始化时间窗口
        time.sleep(1.5) 
        
        # 构造并下发双轴同步复位指令
        cmd = f"X{center_signal}Y{center_signal}\n"
        ser.write(cmd.encode('utf-8'))
        
        print(f"SYSTEM_INFO: Hardware zero-point reset command sent -> {cmd.strip()}")
        
        # 释放系统串口资源
        ser.close()
        
    except serial.SerialException as e:
        print(f"FATAL_ERROR: Serial communication failed. Hardware disconnected or port occupied.\nDetails: {e}")

if __name__ == "__main__":
    reset_servos()