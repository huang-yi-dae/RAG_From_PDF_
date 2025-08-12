import psutil
import time
import os
import platform
import subprocess
import re
from datetime import datetime

try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

def is_wsl():
    """检测是否运行在WSL环境中"""
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
            return 'microsoft' in version.lower()
    except:
        return False

def clear_screen():
    """清除终端屏幕"""
    os_name = platform.system()
    if os_name == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

def get_cpu_usage():
    """获取CPU使用率信息"""
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    total_percent = sum(cpu_percent) / len(cpu_percent)
    return {
        'total': total_percent,
        'cores': cpu_percent,
        'count': len(cpu_percent)
    }

def get_memory_usage():
    """获取内存使用率信息"""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'percent': memory.percent
    }

def get_wsl_gpu_usage():
    """在WSL环境中通过nvidia-smi获取GPU信息"""
    try:
        # 运行nvidia-smi命令获取GPU信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        output = result.stdout.strip()
        if not output:
            return None
            
        gpu_info = []
        for line in output.split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpu_info.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'load': float(parts[2]),
                    'memory_used': float(parts[3]),
                    'memory_total': float(parts[4]),
                    'memory_percent': (float(parts[3]) / float(parts[4])) * 100 if float(parts[4]) > 0 else 0
                })
        return gpu_info
    except Exception as e:
        print(f"获取WSL GPU信息出错: {e}")
        return None

def get_gpu_usage():
    """获取GPU使用率信息，优先考虑WSL环境"""
    if is_wsl():
        # 对于WSL环境，优先使用nvidia-smi
        wsl_gpu = get_wsl_gpu_usage()
        if wsl_gpu is not None:
            return wsl_gpu
    
    # 非WSL环境或WSL中nvidia-smi不可用时，使用GPUtil
    if not gpu_available:
        return None
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,  # 转换为百分比
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100
            })
        return gpu_info
    except Exception as e:
        print(f"获取GPU信息出错: {e}")
        return None

def format_size(bytes, suffix='B'):
    """将字节转换为人类可读的格式"""
    factor = 1024
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < factor:
            return f"{bytes:.2f} {unit}{suffix}"
        bytes /= factor

def print_monitor_data(cpu, memory, gpu, update_interval):
    """打印监控数据到控制台"""
    clear_screen()
    
    # 显示系统环境信息
    env_info = f"环境: {'WSL' if is_wsl() else platform.system()}"
    print(f"系统资源监控工具 - {env_info} - 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"更新间隔: {update_interval}秒 | 按 Ctrl+C 退出\n")
    
    # 打印CPU信息
    print("="*50)
    print(f"CPU 使用率: {cpu['total']:.2f}%")
    print(f"核心数: {cpu['count']}")
    print("核心使用率:")
    for i, core in enumerate(cpu['cores']):
        bar_length = 20
        filled_length = int(bar_length * core / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"  核心 {i}: [{bar}] {core:.2f}%")
    
    # 打印内存信息
    print("\n" + "="*50)
    print(f"内存使用率: {memory['percent']:.2f}%")
    print(f"总内存: {format_size(memory['total'])}")
    print(f"已使用: {format_size(memory['used'])}")
    print(f"可用: {format_size(memory['available'])}")
    
    # 打印GPU信息（如果可用）
    if gpu:
        print("\n" + "="*50)
        print(f"GPU 数量: {len(gpu)}")
        for g in gpu:
            print(f"\nGPU {g['id']}: {g['name']}")
            print(f"  负载: {g['load']:.2f}%")
            print(f"  内存使用: {g['memory_used']:.2f}MB / {g['memory_total']:.2f}MB ({g['memory_percent']:.2f}%)")
    else:
        print("\n" + "="*50)
        print("未检测到可用GPU或GPU监控不可用")
        if is_wsl():
            print("提示: 确保已安装NVIDIA WSL驱动并启用GPU加速")
    
    print("\n" + "="*50)

def main(update_interval=2):
    """主函数：持续监控并显示系统资源使用情况"""
    print("系统资源监控工具启动中...")
    print(f"检测到环境: {'WSL' if is_wsl() else platform.system()}")
    print(f"更新间隔设置为 {update_interval} 秒")
    print("按 Ctrl+C 退出监控\n")
    time.sleep(1)
    
    try:
        while True:
            cpu = get_cpu_usage()
            memory = get_memory_usage()
            gpu = get_gpu_usage()
            
            print_monitor_data(cpu, memory, gpu, update_interval)
            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("\n监控已停止。感谢使用！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='支持WSL的系统资源监控工具')
    parser.add_argument('-i', '--interval', type=int, default=2, 
                      help='更新间隔时间（秒），默认2秒')
    args = parser.parse_args()
    
    main(args.interval)
