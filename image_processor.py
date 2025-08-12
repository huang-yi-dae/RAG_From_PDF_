import os
import json
import glob
import time
import psutil
import subprocess
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path
import io
import base64
import requests
import pytesseract
from datetime import datetime


# ------------------------------
# 系统资源监控
# ------------------------------
def get_system_stats():
    """获取系统资源使用情况，包括GPU（如果可用）"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        "cpu": cpu_percent,
        "mem_used": round(mem.used / (1024 **3), 2),
        "mem_total": round(mem.total / (1024** 3), 2),
        "mem_percent": mem.percent,
        "disk_used": round(disk.used / (1024 **3), 2),
        "disk_total": round(disk.total / (1024** 3), 2),
        "disk_percent": disk.percent,
        "gpu_usage": None
    }
    
    # 尝试获取NVIDIA GPU使用率（需要nvidia-smi）
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip()
        if result:
            stats["gpu_usage"] = int(result.split('\n')[0])  # 取第一个GPU的使用率
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
        
    return stats


def format_stats(stats):
    """格式化系统状态为简洁字符串，包含GPU信息"""
    base_stats = (f"CPU: {stats['cpu']}% | 内存: {stats['mem_used']}/{stats['mem_total']}GB "
                 f"({stats['mem_percent']}%) | 磁盘: {stats['disk_used']}/{stats['disk_total']}GB")
    
    if stats["gpu_usage"] is not None:
        return f"{base_stats} | GPU: {stats['gpu_usage']}%"
    return base_stats


# ------------------------------
# 服务检查与GPU配置验证
# ------------------------------
def check_ollama_status(ollama_host="http://localhost:11434"):
    """检查Ollama服务状态，验证GPU配置"""
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama服务未响应"
            
        # 检查Ollama是否使用GPU
        try:
            log_path = Path.home() / ".ollama" / "logs" / "server.log"
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    last_lines = f.readlines()[-50:]  # 检查最后50行
                    gpu_detected = any("gpu" in line.lower() and "init" in line.lower() for line in last_lines)
                    if gpu_detected:
                        return True, "Ollama服务正常（GPU已启用）"
                    else:
                        return True, "Ollama服务正常（未检测到GPU使用，请检查配置）"
            else:
                return True, "Ollama服务正常（无法验证GPU使用情况）"
                
        except Exception as e:
            return True, f"Ollama服务正常（GPU检查出错: {str(e)}）"
            
    except Exception as e:
        return False, f"Ollama服务不可用: {str(e)}"


# ------------------------------
# 图片处理与文字提取
# ------------------------------
def improve_image(image_path, target_size=(224, 224)):
    """增强图片质量并返回base64编码"""
    try:
        with Image.open(image_path) as img:
            # 处理透明背景
            if img.mode in ('RGBA', 'LA'):
                background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                background.paste(img, img.split()[-1])
                img = background
            
            # 转换为RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 图像增强处理
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.8)
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.2)
            img = ImageOps.autocontrast(img, cutoff=1)
            
            # 尺寸标准化
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            new_img = Image.new(img.mode, target_size, (255, 255, 255))
            paste_x = (target_size[0] - img.size[0]) // 2
            paste_y = (target_size[1] - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # 转换为base64
            buffer = io.BytesIO()
            new_img.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"图片增强失败 {os.path.basename(image_path)}: {str(e)}")
        raise


def extract_text_from_image(image_path):
    """从图片中提取文字"""
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            custom_config = r'--oem 3 --psm 6 -l chi_sim+eng'
            return pytesseract.image_to_string(img_gray, config=custom_config).strip()
    except Exception as e:
        print(f"图片文字提取失败 {image_path}: {str(e)}")
        return ""


def filter_content(content, content_type, min_char_count=10, json_file=None):
    """筛选有效内容"""
    try:
        if not content:
            reason = f"内容为空（类型: {content_type}）"
            return False, reason

        if content_type == "image":
            if not isinstance(content, str) or not os.path.exists(content):
                reason = f"图片路径无效: {str(content)[:50]}"
                return False, reason
            
            text = extract_text_from_image(content)
            source = f"图片 {os.path.basename(content)}"
        elif content_type == "table":
            text = str(content)
            source = "表格内容"
        else:
            reason = f"不支持的类型: {content_type}"
            return False, reason

        # 计算有效字符数（排除空白字符）
        valid_chars = [c for c in text if c.strip()]
        char_count = len(valid_chars)
        
        if char_count >= min_char_count:
            reason = f"{source} 有效字符数达标: {char_count}/{min_char_count}"
            return True, reason
        else:
            reason = f"{source} 有效字符不足: {char_count}/{min_char_count}"
            return False, reason
            
    except Exception as e:
        return False, f"筛选出错: {str(e)}"


# ------------------------------
# Prompt匹配与模型调用
# ------------------------------
def get_matched_prompt(image_text):
    """根据图片文字匹配最佳Prompt"""
    scene_templates = {
        "财务数据": """分析以下财务数据图表:
1. 提取所有关键指标名称和数值
2. 指出数据的时间范围和统计周期
3. 识别显著的增长或下降趋势(超过10%)
4. 用结构化格式呈现结果""",
        
        "趋势图表": """分析以下趋势图表:
1. 确定核心指标和时间范围
2. 描述整体趋势走向
3. 指出关键拐点及其可能原因
4. 用结构化格式呈现时间点和对应数值""",
        
        "市场分析": """分析以下市场分析图表:
1. 识别参与对比的市场主体
2. 提取各主体的市场份额或关键指标
3. 分析竞争格局和差距
4. 用结构化格式呈现对比结果""",
        
        "通用图表": """分析以下图表:
1. 说明图表类型和核心主题
2. 提取所有可见的数据和标签
3. 总结图表传达的关键信息
4. 用清晰的结构化格式呈现结果"""
    }
    
    # 关键词匹配
    if any(kw in image_text for kw in ["营收", "利润", "成本", "财务", "报表", "毛利率"]):
        return "财务数据", scene_templates["财务数据"]
    elif any(kw in image_text for kw in ["趋势", "走势", "变化", "增长率", "周期"]):
        return "趋势图表", scene_templates["趋势图表"]
    elif any(kw in image_text for kw in ["市场", "份额", "竞争", "对比", "占比"]):
        return "市场分析", scene_templates["市场分析"]
    else:
        return "通用图表", scene_templates["通用图表"]


def analyze_image_with_model(image_path, enhanced_base64, **kwargs):
    """调用Ollama分析图片"""
    try:
        image_text = extract_text_from_image(image_path)
        scene, prompt = get_matched_prompt(image_text)
        
        if kwargs.get('verbose', False):
            print(f"  匹配场景: {scene}")

        ollama_host = kwargs.get("ollama_host", "http://localhost:11434").rstrip('/')
        api_url = f"{ollama_host}/api/generate"
        
        # 检查Ollama状态和GPU配置
        status, msg = check_ollama_status(ollama_host)
        if not status:
            raise ConnectionError(f"Ollama服务问题: {msg}")
        if kwargs.get('verbose', False):
            print(f"  Ollama状态: {msg}")

        # Ollama支持的参数配置
        options = {
            "num_gpu": kwargs.get("num_gpu", -1),  # -1表示使用所有可用GPU
            "temperature": 0.1,
            "num_predict": 2048,
            "top_k": 30,
            "top_p": 0.9
        }

        if kwargs.get('verbose', False):
            print(f"  调用Ollama API: {api_url}")
            print(f"  使用模型: {kwargs.get('model', 'qwen2.5-vl')}")
            print(f"  配置参数: {options}")

        response = requests.post(
            api_url,
            json={
                "model": kwargs.get("model", "qwen2.5-vl"),
                "prompt": prompt,
                "images": [enhanced_base64],
                "stream": False,
                "options": options
            },
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API返回错误状态: {response.status_code}, 内容: {response.text}")
        
        response_data = response.json()
        if "response" not in response_data:
            raise Exception(f"Ollama API返回格式不正确: {response_data}")
            
        return response_data["response"]
        
    except Exception as e:
        raise Exception(f"图片分析失败: {str(e)}")


# ------------------------------
# 主处理函数（加入筛选逻辑）
# ------------------------------
def process_json_files(input_folder, output_folder,** kwargs):
    """处理JSON文件，加入内容筛选逻辑"""
    auto_delete = kwargs.get('auto_delete_processed', False)
    verbose = kwargs.get('verbose', False)
    progress_file = kwargs.get("progress_file", "processing_progress.json")
    stats_update_interval = kwargs.get('stats_interval', 3)
    min_char_count = kwargs.get('min_char_count', 10)  # 从配置获取文字数量阈值
    
    # 加载进度
    processed = load_processed_files(progress_file)
    processed_files = processed.get("processed_files", [])

    # 获取待处理文件
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    pending_files = [f for f in json_files if f not in processed_files]
    total_files = len(pending_files)
    
    if total_files == 0:
        print("没有待处理文件")
        return processed_files

    # 初始化进度条
    total_pbar = tqdm(total=total_files, desc="总进度")
    last_stats_update = 0
    stats = get_system_stats()
    total_pbar.set_postfix_str(format_stats(stats))

    # 处理统计
    processing_stats = {
        "success": 0,
        "failed": 0,
        "filtered": 0,  # 新增：筛选未通过的数量
        "skipped": 0,
        "start_time": datetime.now()
    }

    for idx, json_file in enumerate(pending_files, 1):
        try:
            filename = os.path.basename(json_file)
            current_time = time.time()
            
            # 定期更新系统状态
            if current_time - last_stats_update > stats_update_interval:
                stats = get_system_stats()
                total_pbar.set_postfix_str(format_stats(stats))
                last_stats_update = current_time

            if verbose:
                print(f"\n[{idx}/{total_files}] 处理文件: {filename}")
            else:
                total_pbar.set_description(f"处理中: {filename[:20]}...")

            # 加载文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 处理列表类型
            if isinstance(data, list):
                original_count = len(data)
                # 倒序处理条目
                for i in range(len(data)-1, -1, -1):
                    item = data[i]
                    item_type = item.get("type")
                    
                    if item_type == "image":
                        image_path = item.get("content", "")
                        
                        # 关键修改：添加内容筛选步骤
                        is_valid, reason = filter_content(
                            content=image_path,
                            content_type="image",
                            min_char_count=min_char_count,
                            json_file=json_file
                        )
                        
                        if verbose:
                            print(f"  {reason}")
                            
                        # 如果筛选不通过，设置content为False并保留图片路径
                        if not is_valid:
                            item["content"] = False
                            item["image_path"] = image_path
                            item["filtered"] = True
                            item["filter_reason"] = reason
                            processing_stats["filtered"] += 1
                            continue
                        
                        # 筛选通过，进行正常处理
                        item["filtered"] = False
                        
                        # 增强图片
                        if verbose:
                            print(f"  增强图片: {os.path.basename(image_path)}")
                        enhanced_base64 = improve_image(image_path)
                        
                        # 分析图片
                        analysis_result = analyze_image_with_model(
                            image_path, enhanced_base64, **kwargs
                        )
                        
                        # 更新条目
                        item["content"] = analysis_result
                        item["image_path"] = image_path
                        item["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                processing_stats["success"] += 1
                if verbose:
                    print(f"  处理完成，原始条目: {original_count}, 保留: {len(data)}, "
                          f"筛选未通过: {processing_stats['filtered']}")

            # 保存结果
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 自动删除选项
            if auto_delete:
                try:
                    os.remove(json_file)
                    if verbose:
                        print(f"  已删除原始文件: {filename}")
                except Exception as e:
                    print(f"  删除文件失败 {filename}: {str(e)}")

            # 更新进度
            processed_files.append(json_file)
            if idx % 10 == 0:
                save_processed_files(progress_file, processed_files)

            total_pbar.update(1)

        except Exception as e:
            processing_stats["failed"] += 1
            if verbose:
                print(f"  处理失败 {filename}: {str(e)}")
            continue

    # 完成处理
    total_pbar.close()
    save_processed_files(progress_file, processed_files)
    
    # 显示汇总统计（包含筛选信息）
    elapsed = (datetime.now() - processing_stats["start_time"]).total_seconds()
    print("\n" + "="*50)
    print(f"处理完成 | 总耗时: {elapsed:.1f}秒")
    print(f"成功: {processing_stats['success']} | 失败: {processing_stats['failed']} | "
          f"筛选未通过: {processing_stats['filtered']} | 跳过: {processing_stats['skipped']}")
    print(f"系统状态: {format_stats(get_system_stats())}")
    print("="*50 + "\n")

    return processed_files


# ------------------------------
# 进度管理
# ------------------------------
def load_processed_files(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载进度文件失败: {str(e)}，将重新创建")
    return {"processed_files": []}


def save_processed_files(progress_file, processed_files):
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_files": processed_files,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存进度失败: {str(e)}")


# ------------------------------
# 主函数（配置筛选阈值）
# ------------------------------
def main():
    # 所有参数在此处直接配置，包括筛选阈值
    config = {
        # 路径配置
        "input": r"/home/baobaodae/Develop/RAG_Data/Version1.0/test/output/",
        "output": r"/home/baobaodae/Develop/RAG_Data/Version1.0/test/output_image_processed",
        
        # Ollama与GPU配置
        "use_ollama": True,
        "ollama_host": "http://localhost:11434",
        "model": "llama3:latest",
        "num_gpu": -1,          # -1表示使用所有GPU
        
        # 内容筛选配置
        "min_char_count": 10,   # 文字数量阈值，低于此值的图片不处理
        
        # 处理配置
        "progress_file": "processing_progress.json",
        "delete_progress_file": True,
        "auto_delete_processed": True,
        "verbose": False,        # 详细模式显示筛选信息
        "stats_interval": 1     # 系统状态更新间隔（秒）
    }

    # 自动处理进度文件（根据配置）
    if config["delete_progress_file"] and os.path.exists(config["progress_file"]):
        try:
            os.remove(config["progress_file"])
            print(f"已删除进度文件: {config['progress_file']}")
        except Exception as e:
            print(f"删除进度文件失败: {str(e)}")

    # 初始化信息
    print("="*50)
    print("          带内容筛选的图片分析处理器")
    print("="*50)
    print("正在初始化...")

    # 验证输入路径
    input_path = Path(config["input"])
    if not input_path.is_dir():
        print(f"错误: 输入路径无效: {input_path}")
        return

    # 创建输出目录
    output_path = Path(config["output"])
    output_path.mkdir(parents=True, exist_ok=True)

    # 检查Ollama状态和GPU配置
    if config["use_ollama"]:
        status, msg = check_ollama_status(config["ollama_host"])
        if not status:
            print(f"错误: {msg}")
            print("程序将退出")
            return
        print(f"Ollama状态: {msg}")

    # 显示配置信息（包含筛选阈值）
    print(f"\n===== 配置信息 =====")
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"Ollama服务: {config['ollama_host']}")
    print(f"使用模型: {config['model']}")
    print(f"GPU配置: 使用 {config['num_gpu'] if config['num_gpu'] != -1 else '所有'} 个GPU")
    print(f"内容筛选: 文字数量阈值 {config['min_char_count']} 个字符")
    print(f"删除进度文件: {'是' if config['delete_progress_file'] else '否'}")
    print(f"自动删除原始文件: {'是' if config['auto_delete_processed'] else '否'}")
    print(f"详细输出模式: {'是' if config['verbose'] else '否'}")
    print(f"====================\n")
    print("开始处理文件...\n")

    # 处理所有文件夹
    folders = [f for f in input_path.rglob('*') if f.is_dir()]
    folder_pbar = tqdm(folders, desc="文件夹处理")
    
    for folder in folder_pbar:
        folder_output = output_path / folder.relative_to(input_path)
        folder_output.mkdir(parents=True, exist_ok=True)
        folder_pbar.set_postfix_str(f"当前: {folder.name}")
        
        process_json_files(
            input_folder=str(folder),
            output_folder=str(folder_output),** config
        )

    folder_pbar.close()
    print("所有处理完成！")


if __name__ == '__main__':
    main()
    