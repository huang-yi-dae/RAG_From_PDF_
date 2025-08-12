import os
import json
import re
import time
import requests
from datetime import datetime
from tqdm import tqdm
from fuzzywuzzy import fuzz
import jieba
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from textrank4zh import TextRank4Keyword


# --------------------------
# 模型调用相关函数
# --------------------------
def call_ollama_model(prompt: str, params: dict) -> str:
    """调用Ollama模型进行主题分类"""
    try:
        data = {
            "model": params["model_name"],
            "prompt": prompt,
            "stream": params["stream"],
            "timeout": params["timeout"] * 1000,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"]
        }
        
        response = requests.post(
            f"{params['host']}/api/generate",
            json=data,
            timeout=params["timeout"]
        )
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            raise Exception(f"Ollama调用失败，状态码: {response.status_code}")
    except Exception as e:
        raise Exception(f"Ollama调用出错: {str(e)}")

def call_api_model(prompt: str, params: dict) -> str:
    """调用外部API进行主题分类"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {params['api_key']}"
        }
        
        data = {
            "model": params["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": params["top_p"],
            "frequency_penalty": params["frequency_penalty"],
            "presence_penalty": params["presence_penalty"]
        }
        
        response = requests.post(
            params["api_url"],
            json=data,
            headers=headers,
            timeout=params["timeout"]
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"API调用失败，状态码: {response.status_code}")
    except Exception as e:
        raise Exception(f"API调用出错: {str(e)}")


# --------------------------
# 传统方法工具函数
# --------------------------
def traditional_tfidf_extract(text: str, params: dict, stopwords: list) -> list:
    """使用TF-IDF提取关键词"""
    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        min_df=params["min_df"],
        max_df=params["max_df"],
        stop_words=stopwords
    )
    try:
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        weights = X.toarray()[0]
        sorted_indices = weights.argsort()[::-1]
        return [feature_names[i] for i in sorted_indices if weights[i] > 0][:10]
    except:
        return []

def traditional_textrank_extract(text: str, params: dict) -> list:
    """使用TextRank提取关键词"""
    if len(text) < params["min_text_length"]:
        return []
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=params["lower"], window=params["window"])
    return [item.word for item in tr4w.get_keywords(10, word_min_len=params["word_min_len"])]


# --------------------------
# 核心分类工具函数
# --------------------------
def load_topic_hierarchy(topic_file_path: str) -> dict:
    """加载主题分类体系，从topic_tree字段获取"""
    try:
        with open(topic_file_path, 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
        
        # 从topic_tree字段获取主题体系，而非原来的topic_hierarchy
        topic_tree = topic_data.get("topic_tree", {})
        
        # 验证主题体系格式
        if not isinstance(topic_tree, dict):
            raise ValueError("主题体系格式错误，应为字典类型")
            
        # 过滤无效的主题条目
        valid_topics = {}
        for main_topic, subtopics in topic_tree.items():
            if isinstance(subtopics, list) and all(isinstance(t, str) for t in subtopics):
                valid_topics[main_topic] = subtopics
            else:
                print(f"过滤无效的主题条目: {main_topic}，子主题格式不正确")
                
        return valid_topics
    except Exception as e:
        print(f"加载主题体系失败: {str(e)}")
        return {}

def get_json_files(json_dir: str) -> list:
    """获取目录下所有JSON文件路径"""
    json_files = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def load_single_json(file_path: str) -> list:
    """加载单个JSON文件内容，记录文件名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        
        # 为每个条目添加文件名元数据
        filename = os.path.basename(file_path)
        for item in data:
            if isinstance(item, dict) and "metadata" not in item:
                item["metadata"] = {}
            item["metadata"]["file_name"] = filename
        
        return [
            item for item in data 
            if isinstance(item, dict) 
            and item.get("type") == "text"
        ]
    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 失败: {str(e)}")
        return []

def clean_text(text: str) -> str:
    """清洗文本"""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_long_text(text: str, max_length: int = 500) -> list:
    """将长文本拆分为多个部分"""
    if len(text) <= max_length:
        return [text]
    
    paragraphs = re.split(r'[。？！；,.?!;]', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_length:
            current_chunk += para + "。"
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para + "。"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_keywords(text: str, config: dict) -> list:
    """根据配置提取关键词（多方法支持）"""
    model_type = config["model"]["type"].lower()
    stopwords = config["stopwords"]
    
    if model_type in ["ollama", "api"]:
        try:
            prompt = f"请从以下文本中提取最多10个核心关键词，用逗号分隔，不要解释：{text[:500]}"
            if model_type == "ollama":
                response = call_ollama_model(prompt, config["model"]["params"])
            else:
                response = call_api_model(prompt, config["model"]["params"])
            if response:
                return [kw.strip() for kw in response.split(',') if kw.strip() and kw not in stopwords]
        except Exception as e:
            print(f"模型提取关键词失败，使用备用方法：{str(e)}")
    
    # 传统方法备用
    tfidf_kw = traditional_tfidf_extract(text, config["traditional_method"]["tfidf"], stopwords)
    textrank_kw = traditional_textrank_extract(text, config["traditional_method"]["textrank"])
    combined = list(set(tfidf_kw + textrank_kw))
    return combined[:10]

def classify_content(text: str, topic_hierarchy: dict, config: dict) -> list:
    """多方法内容分类"""
    if not text or not topic_hierarchy:
        return []
    
    model_type = config["model"]["type"].lower()
    
    # 生成规范化主题列表
    topic_list = []
    for main_topic, subtopics in topic_hierarchy.items():
        # 添加主主题
        topic_list.append(main_topic)
        # 添加子主题（与主主题关联）
        if isinstance(subtopics, list):
            for sub in subtopics:
                topic_list.append(f"{main_topic} > {sub}")
    
    # 主题列表格式化（编号+换行），增强模型识别度
    topic_str = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topic_list)])
    
    # 使用模型进行分类
    if model_type in ["ollama", "api"]:
        try:
            prompt = f"""请严格按照以下规则为文本匹配主题：
1. 仅从【主题列表】中选择最相关的3个主题，**绝对禁止使用列表外的任何主题**。
2. 主题格式：
   - 若为主主题：直接写主题名（如"金融产品与市场"）
   - 若为子主题：主主题 > 子主题（如"金融产品与市场 > 产品特性"）
3. 输出要求：只返回主题本身，**不添加任何解释、标点外的文字**，用英文逗号分隔。
4. 无匹配主题时，**仅返回"未分类"**。

【主题列表】：
{topic_str}

【文本】：
{text[:500]}
"""
            if model_type == "ollama":
                response = call_ollama_model(prompt, config["model"]["params"])
            else:
                response = call_api_model(prompt, config["model"]["params"])
            
            if response and "未分类" not in response:
                results = []
                for item in response.split(','):
                    item = item.strip()
                    # 严格校验主题是否在列表中
                    if item in topic_list:
                        if " > " in item:
                            main, sub = item.split(" > ", 1)
                            results.append((main.strip(), sub.strip()))
                        else:
                            results.append((item, ""))
                return list(set(results))[:3]  # 去重
        except Exception as e:
            print(f"模型分类失败，使用传统方法：{str(e)}")
    
    # 传统分类方法
    keywords = extract_keywords(text, config)
    if not keywords:
        return []
    
    topic_scores = defaultdict(int)
    min_similarity = config["classification"]["min_similarity"]
    valid_topics = [(t, t.split(" > ")) for t in topic_list]
    
    for kw in keywords:
        for full_topic, (main_topic, sub_topic) in valid_topics:
            # 计算关键词与主题的相似度
            similarity = fuzz.ratio(kw.lower(), full_topic.lower())
            if similarity >= min_similarity:
                topic_scores[(main_topic, sub_topic[0] if len(sub_topic)>1 else "")] += similarity
    
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [(topic[0], topic[1]) for topic, score in sorted_topics if score > 0]

def process_content(content: str, item: dict, topic_hierarchy: dict, config: dict) -> list:
    """处理单个内容，保留page和file_name元数据"""
    if content is False:
        return []
    
    # 提取原始元数据（page和file_name）
    item_id = item.get("id", f"unknown_{hash(str(content)[:10])}")
    metadata = item.get("metadata", {})
    page = metadata.get("page")  # 从元数据获取页码
    file_name = metadata.get("file_name")  # 获取文件名
    
    # 转换为字符串并清洗
    if not isinstance(content, str):
        content = str(content)
    cleaned_content = clean_text(content)
    
    # 拆分长文本
    text_chunks = split_long_text(cleaned_content, config["processing"]["max_text_length"])
    
    processed_chunks = []
    for i, chunk in enumerate(text_chunks):
        # 生成子ID
        chunk_id = f"{item_id}_chunk_{i+1}" if len(text_chunks) > 1 else item_id
        
        # 分类
        topics = classify_content(chunk, topic_hierarchy, config)
        is_unclassified = len(topics) == 0
        
        # 保留原始元数据并添加处理信息
        chunk_metadata = {
            "file_name": file_name,
            "page": page,
            "original_id": item_id,
            "chunk_index": i + 1 if len(text_chunks) > 1 else None,
            "total_chunks": len(text_chunks) if len(text_chunks) > 1 else None
        }
        
        processed_chunks.append({
            "id": chunk_id,
            "content": chunk,
            "topics": topics,
            "is_unclassified": is_unclassified,
            "metadata": chunk_metadata  # 集中存储元数据
        })
    
    return processed_chunks


# --------------------------
# 主函数
# --------------------------
def main():
    # 配置（保留元数据相关设置）
    config = {
        "paths": {
            "json_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/output_image_processed/elements_jsons",
            "topic_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/config_updates/updated_config.json",
            "output_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_contents",
        },
        "processing": {
            "max_text_length": 500,  # 文本拆分的最大长度
            "max_retries": 2,         # 模型调用重试次数
        },
        "model": {
            "type": "ollama",  # "ollama" / "api" / "traditional"
            "params": {
                # Ollama参数
                "host": "http://localhost:11434",
                "model_name": "llama3:latest",
                "stream": False,
                "timeout": 300,
                "temperature": 0.1,  # 降低温度，减少创造性输出
                "top_p": 0.9,
                "top_k": 40,
                
                # API参数
                "api_url": "https://api.siliconflow.cn/v1/chat/completions",
                "api_key": "",
                "max_tokens": 200,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        },
        "traditional_method": {
            "tfidf": {
                "max_features": 1000,
                "min_df": 1,
                "max_df": 0.95
            },
            "textrank": {
                "window": 2,
                "lower": True,
                "word_min_len": 2,
                "min_text_length": 10
            }
        },
        "classification": {
            "min_similarity": 60,  # 传统方法最小相似度阈值
            "unclassified_label": "未分类"  # 未分类标签名称
        },
        "stopwords": [
            "证", "券", "研", "究", "报", "告", "公", "司", "页", 
            "资料来源", "wind", "同花顺", "东方财富", "日期", "单位"
        ]
    }
    
    # 创建输出目录
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    
    # 加载主题体系
    print("加载主题分类体系...")
    topic_hierarchy = load_topic_hierarchy(config["paths"]["topic_file"])
    if not topic_hierarchy:
        print("未找到有效的主题分类体系，程序终止")
        return
    
    # 显示加载的主题体系
    print(f"成功加载主题体系，包含 {len(topic_hierarchy)} 个主主题:")
    for main_topic, subtopics in topic_hierarchy.items():
        print(f"- {main_topic}: 包含 {len(subtopics)} 个子主题")
    
    # 获取所有JSON文件
    print("查找所有JSON文件...")
    json_files = get_json_files(config["paths"]["json_dir"])
    if not json_files:
        print("未找到任何JSON文件，程序终止")
        return
    print(f"共发现 {len(json_files)} 个JSON文件")
    
    # 初始化结果存储
    all_classified = []
    label_mapping = defaultdict(list)  # 标签到内容ID的映射
    unclassified_ids = []  # 未分类内容ID列表
    
    # 处理每个文件
    for file_path in tqdm(json_files, desc="处理文件"):
        try:
            items = load_single_json(file_path)  # 加载时已添加file_name
            if not items:
                continue
            
            for item in items:
                content = item.get("content")
                
                # 处理内容（传入完整item以提取元数据）
                processed = process_content(
                    content, 
                    item, 
                    topic_hierarchy,
                    config
                )
                
                if processed:
                    all_classified.extend(processed)
                    
                    # 更新标签映射和未分类列表
                    for chunk in processed:
                        if chunk["is_unclassified"]:
                            unclassified_ids.append(chunk["id"])
                        else:
                            for main_topic, sub_topic in chunk["topics"]:
                                label = f"{main_topic} > {sub_topic}" if sub_topic else main_topic
                                label_mapping[label].append(chunk["id"])
        
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
            continue
    
    # 单独添加未分类标签映射
    unclassified_label = config["classification"]["unclassified_label"]
    label_mapping[unclassified_label] = sorted(list(set(unclassified_ids)))
    
    # 保存分类结果（含page和file_name）
    classified_output = os.path.join(config["paths"]["output_dir"], "classified_contents.json")
    with open(classified_output, 'w', encoding='utf-8') as f:
        json.dump(all_classified, f, ensure_ascii=False, indent=2)
    print(f"分类结果已保存至: {classified_output}")
    
    # 保存标签映射（含未分类）
    mapping_output = os.path.join(config["paths"]["output_dir"], "label_mapping.json")
    for label in label_mapping:
        label_mapping[label] = sorted(list(set(label_mapping[label])))
    
    with open(mapping_output, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    print(f"标签映射已保存至: {mapping_output}")
    
    # 统计信息
    total = len(all_classified)
    unclassified_count = len(unclassified_ids)
    classified_count = total - unclassified_count
    print(f"\n处理完成：")
    print(f"总内容片段：{total}")
    print(f"已分类：{classified_count}（{classified_count/total*100:.2f}%）")
    print(f"未分类：{unclassified_count}（{unclassified_count/total*100:.2f}%）")
    print(f"标签总数：{len(label_mapping)}（含未分类标签）")

if __name__ == "__main__":
    main()
