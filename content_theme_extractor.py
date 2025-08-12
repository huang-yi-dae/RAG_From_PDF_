import json
import re
import os
import time
import requests
from datetime import datetime
from tqdm import tqdm
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textrank4zh import TextRank4Keyword
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import signal
from functools import wraps
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# --------------------------
# 超时处理装饰器
# --------------------------
class TimeoutException(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"函数执行超时，超过{seconds}秒")
            
            # 设置信号处理（Windows系统可能需要调整）
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)  # 触发超时
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # 重置超时
                return result
            except TimeoutException as e:
                raise e
            finally:
                signal.alarm(0)  # 确保超时被重置
        return wrapper
    return decorator


# --------------------------
# 模型调用相关函数
# --------------------------
def check_ollama_status(ollama_host: str = "http://localhost:11434") -> bool:
    """检查Ollama服务是否正在运行"""
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False

def call_ollama_model(prompt: str, params: Dict) -> str:
    """调用Ollama模型生成关键词（使用明确参数）"""
    try:
        data = {
            "model": params["model_name"],
            "prompt": prompt,
            "stream": params["stream"],
            "timeout": params["timeout"] * 1000,  # 转换为毫秒
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
            result = response.json()
            return result.get("response", "").strip()
        else:
            raise Exception(f"Ollama调用失败，状态码: {response.status_code}")
    except Exception as e:
        raise Exception(f"Ollama调用出错: {str(e)}")

def call_api_model(prompt: str, params: Dict) -> str:
    """调用外部API生成关键词（使用明确参数）"""
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
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"API调用失败，状态码: {response.status_code}, 响应: {response.text[:200]}")
    except Exception as e:
        raise Exception(f"API调用出错: {str(e)}")


# --------------------------
# 工具函数：进度记录与管理
# --------------------------
def load_progress(progress_path: str) -> Dict:
    """加载进度，确保所有必要键存在"""
    progress_template = {
        "total_files": 0,
        "processed_files": [],
        "failed_files": [],
        "processing_files": [],
        "completed_extraction": False,
        "completed_analysis": False,
        "completed_updating": False,  # 新增：标记自动更新是否完成
        "start_time": "",
        "last_update_time": "",
        "total_duration": 0
    }
    
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                loaded_progress = json.load(f)
            for key in progress_template:
                if key not in loaded_progress:
                    loaded_progress[key] = progress_template[key]
            return loaded_progress
        except (json.JSONDecodeError, KeyError):
            print(f"进度文件损坏，将创建新的进度文件：{progress_path}")
    
    return progress_template.copy()

def save_progress(progress_path: str, progress: Dict) -> None:
    """保存处理进度，包含时间戳"""
    now = datetime.now()
    progress["last_update_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
    if not progress["start_time"]:
        progress["start_time"] = progress["last_update_time"]
    else:
        try:
            start_time = datetime.strptime(progress["start_time"], "%Y-%m-%d %H:%M:%S")
            progress["total_duration"] = int((now - start_time).total_seconds())
        except ValueError:
            progress["total_duration"] = 0
    
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def delete_progress(progress_path: str) -> None:
    """删除进度文件"""
    if os.path.exists(progress_path):
        try:
            os.remove(progress_path)
            print(f"已删除进度文件：{progress_path}")
        except Exception as e:
            print(f"删除进度文件失败：{str(e)}")

def print_progress_summary(progress: Dict) -> None:
    """打印进度摘要"""
    total = progress.get("total_files", 0)
    processed = len(progress.get("processed_files", []))
    failed = len(progress.get("failed_files", []))
    processing = len(progress.get("processing_files", []))
    pending = max(0, total - processed - failed - processing)
    
    total_duration = progress.get("total_duration", 0)
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}时{minutes}分{seconds}秒"
    
    print("\n" + "="*50)
    print(f"处理进度摘要：")
    print(f"总文件数：{total}")
    print(f"已完成：{processed} ({processed/total*100:.2f}%)" if total > 0 else f"已完成：{processed}")
    print(f"处理中：{processing}")
    print(f"待处理：{pending}")
    print(f"处理失败：{failed} ({failed/total*100:.2f}%)" if total > 0 else f"处理失败：{failed}")
    print(f"主题词提取完成：{'是' if progress.get('completed_extraction', False) else '否'}")
    print(f"主题关系分析完成：{'是' if progress.get('completed_analysis', False) else '否'}")
    print(f"词典自动更新完成：{'是' if progress.get('completed_updating', False) else '否'}")
    print(f"开始时间：{progress.get('start_time', '未记录')}")
    print(f"最后更新：{progress.get('last_update_time', '未记录')}")
    print(f"总处理时间：{duration_str}")
    print("="*50 + "\n")


# --------------------------
# 数据加载与预处理
# --------------------------
def get_json_files(json_dir: str) -> List[str]:
    """获取目录下所有JSON文件路径（支持嵌套目录）"""
    json_files = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def load_single_json(file_path: str) -> List[Dict]:
    """加载单个JSON文件内容（容错处理）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        return [
            item for item in data 
            if isinstance(item, dict) 
            and item.get("type") == "text" 
            and item.get("content") is not False
        ]
    except Exception as e:
        raise Exception(f"加载文件失败 {os.path.basename(file_path)}：{str(e)}")

def clean_text(text: str, stopwords: List[str]) -> str:
    """清洗文本"""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,.?!;：，。？！；]', ' ', text)
    if stopwords:
        stop_pattern = re.compile(r'\b(' + '|'.join(re.escape(sw) for sw in stopwords) + r')\b')
        text = stop_pattern.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


# --------------------------
# 分词与领域增强
# --------------------------
def init_jieba(financial_dict: Dict) -> None:
    """初始化jieba，加载金融领域词典（去重处理）"""
    all_terms = list({term for terms in financial_dict.values() for term in terms if isinstance(term, str)})
    for term in all_terms:
        jieba.add_word(term)
    fixed_terms = [("伏美替尼"), ("非小细胞肺癌")]
    for term in fixed_terms:
        jieba.suggest_freq(term, True)


# --------------------------
# 主题词提取（支持多种模型调用方式）
# --------------------------
@timeout(60)
def extract_keywords(text: str, words: List[str], config: Dict) -> List[str]:
    """提取关键词（使用明确参数）"""
    top_k = config["topic_extraction"]["top_k_keywords"]
    stopwords = config["stopwords"]
    model_type = config["model"]["type"].lower()
    
    # 基础领域词提取
    domain_terms = list({w for w in words if w in [t for ts in config["financial_dict"].values() for t in ts]})
    
    # 调用Ollama模型
    if model_type == "ollama":
        try:
            prompt = f"""请从以下金融文本中提取最多{top_k}个核心关键词，仅返回关键词，用逗号分隔，不要解释：
文本：{text[:500]}
领域参考词：{','.join(domain_terms[:5])}
"""
            response = call_ollama_model(prompt, config["model"]["params"])
            if response:
                keywords = [kw.strip() for kw in response.split(',') if kw.strip() and kw.strip() not in stopwords]
                return list(dict.fromkeys(keywords))[:top_k]
            else:
                print("Ollama返回空结果，使用备用方法")
        except Exception as e:
            print(f"Ollama调用失败: {str(e)}，使用备用方法")
    
    # 调用API模型
    elif model_type == "api":
        try:
            prompt = f"""请从以下金融文本中提取最多{top_k}个核心关键词，仅返回关键词，用逗号分隔，不要解释：
文本：{text[:500]}
领域参考词：{','.join(domain_terms[:5])}
"""
            response = call_api_model(prompt, config["model"]["params"])
            if response:
                keywords = [kw.strip() for kw in response.split(',') if kw.strip() and kw.strip() not in stopwords]
                return list(dict.fromkeys(keywords))[:top_k]
            else:
                print("API返回空结果，使用备用方法")
        except Exception as e:
            print(f"API调用失败: {str(e)}，使用备用方法")
    
    # 传统方法（TF-IDF + TextRank）
    corpus = [" ".join(words)]
    tfidf_params = config["traditional_method"]["tfidf"]
    tfidf_kw = extract_tfidf_keywords(corpus, top_k, stopwords, tfidf_params)[0]
    
    textrank_params = config["traditional_method"]["textrank"]
    textrank_kw = extract_textrank_keywords(text, top_k, textrank_params)
    
    return merge_keywords(tfidf_kw, textrank_kw, domain_terms, config["financial_dict"], top_k)

def extract_tfidf_keywords(corpus: List[str], top_k: int, stopwords: List[str], params: Dict) -> List[List[str]]:
    """TF-IDF提取关键词（使用明确参数）"""
    if not corpus or all(not doc.strip() for doc in corpus):
        return [[] for _ in corpus]
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(),
        max_features=params["max_features"],
        min_df=params["min_df"],
        max_df=params["max_df"],
        stop_words=stopwords
    )
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError:
        return [[] for _ in corpus]
    feature_names = vectorizer.get_feature_names_out()
    keywords_list = []
    for i in range(len(corpus)):
        weights = X[i].toarray()[0]
        indices = weights.argsort()[::-1][:top_k]
        keywords = [feature_names[idx] for idx in indices if weights[idx] > 0]
        keywords_list.append(keywords)
    return keywords_list

def extract_textrank_keywords(text: str, top_k: int, params: Dict) -> List[str]:
    """TextRank提取关键词（使用明确参数）"""
    if len(text) < params["min_text_length"]:
        return []
    tr4w = TextRank4Keyword()
    tr4w.analyze(
        text=text, 
        lower=params["lower"], 
        window=params["window"]
    )
    return [item.word for item in tr4w.get_keywords(top_k, word_min_len=params["word_min_len"])]

def merge_keywords(
    tfidf_kw: List[str], 
    textrank_kw: List[str], 
    domain_terms: List[str], 
    financial_dict: Dict, 
    top_k: int
) -> List[str]:
    """融合多源关键词"""
    all_domain_terms = [term for terms in financial_dict.values() for term in terms if isinstance(term, str)]
    merged = list({kw for kw in tfidf_kw + textrank_kw + domain_terms if kw})
    merged.sort(key=lambda x: 0 if x in all_domain_terms else 1)
    return merged[:top_k]


# --------------------------
# 第一阶段：提取所有主题词并保存到临时文件
# --------------------------
def process_single_file_extraction(
    file_path: str, 
    config: Dict,
    temp_file: str
) -> None:
    """处理单个JSON文件，提取主题词并写入临时文件"""
    start_time = time.time()
    max_file_duration = config["processing"]["max_file_duration"]
    
    try:
        def check_timeout():
            if time.time() - start_time > max_file_duration:
                raise TimeoutException(f"文件处理超时（>{max_file_duration}秒）")
        
        text_items = load_single_json(file_path)
        if not text_items:
            return
        
        for item in text_items:
            check_timeout()
            raw_text = item.get("content", "")
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
                
            cleaned_text = clean_text(raw_text, config["stopwords"])
            if len(cleaned_text) < config["processing"]["min_text_length"]:
                continue
            
            check_timeout()
            topic_segments = split_multi_topic(cleaned_text)
            if not topic_segments:
                topic_segments = [("整体主题", cleaned_text)]
            
            for title, segment in topic_segments:
                check_timeout()
                words = jieba.cut(segment)
                words = [word for word in words if word.strip() and word not in config["stopwords"]]
                if not words:
                    continue
                
                core_kw = extract_keywords(segment, words, config)
                # 过滤掉非关键词的描述性文本
                core_kw = [kw for kw in core_kw if not re.match(r'^[A-Za-z\s]+$', kw) or len(kw.split()) <= 1]
                if not core_kw:
                    continue
                
                with open(temp_file, 'a', encoding='utf-8') as f:
                    entry = {
                        "file_path": file_path,
                        "item_id": item.get("id", f"unknown_{hash(raw_text[:10])}"),
                        "title": title,
                        "core_keywords": core_kw,
                        "segment_length": len(segment)
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
    except TimeoutException as e:
        raise e
    except Exception as e:
        raise Exception(f"处理文件出错：{str(e)}")


def process_batch_extraction(batch_files: List[str], config: Dict, progress: Dict) -> List[str]:
    """处理一个批次的文件，提取主题词"""
    failed_files = []
    max_batch_duration = config["processing"]["max_batch_duration"]
    batch_start_time = time.time()
    temp_file = config["paths"]["temp_keywords_file"]
    
    progress["processing_files"] = list(set((progress["processing_files"])) | set(batch_files))
    save_progress(config["paths"]["progress_file"], progress)
    
    for file_idx, file_path in enumerate(batch_files):
        if time.time() - batch_start_time > max_batch_duration:
            remaining_files = batch_files[file_idx:]
            failed_files.extend(remaining_files)
            print(f"批次超时，剩余{len(remaining_files)}个文件移至下一批")
            break
        
        retries = 0
        success = False
        while retries < config["processing"]["max_retries"] and not success:
            try:
                process_single_file_extraction(file_path, config, temp_file)
                success = True
            except TimeoutException as e:
                print(f"文件超时：{str(e)}")
                retries = config["processing"]["max_retries"]
            except Exception as e:
                retries += 1
                print(f"重试 {retries}/{config['processing']['max_retries']}：{str(e)}")
                if retries >= config["processing"]["max_retries"]:
                    failed_files.append(file_path)
                    tqdm.write(f"文件 {os.path.basename(file_path)} 处理失败")
        
        if file_path in progress["processing_files"]:
            progress["processing_files"].remove(file_path)
        if success:
            progress["processed_files"].append(file_path)
            tqdm.write(f"成功处理：{os.path.basename(file_path)}")
        else:
            progress["failed_files"].append(file_path)
        save_progress(config["paths"]["progress_file"], progress)
    
    return failed_files


# --------------------------
# 第二阶段：从临时文件分析主题关系
# --------------------------
def split_multi_topic(text: str) -> List[Tuple[str, str]]:
    """按报告标题标记拆分多主题段落"""
    if not text:
        return []
    pattern = r'[◼■●]+\s*[\u4e00-\u9fa5a-zA-Z0-9_]+'
    topics = re.split(pattern, text)
    topics = [t.strip() for t in topics if t.strip()]
    titles = re.findall(pattern, text)
    if len(titles) < len(topics):
        titles += [f"主题{i+1}" for i in range(len(topics) - len(titles))]
    return list(zip(titles, topics))

def analyze_topic_relations(config: Dict) -> Dict:
    """从临时关键词文件分析主题及次主题关系"""
    print("\n开始分析主题关系...")
    temp_file = config["paths"]["temp_keywords_file"]
    
    if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
        print("临时关键词文件不存在或为空，无法进行主题分析")
        return {}
    
    all_keywords = []
    keyword_counter = Counter()
    keyword_contexts = defaultdict(list)
    
    with open(temp_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                keywords = entry.get("core_keywords", [])
                if keywords:
                    all_keywords.append(keywords)
                    keyword_counter.update(keywords)
                    for i, kw1 in enumerate(keywords):
                        for j, kw2 in enumerate(keywords):
                            if i < j:
                                keyword_contexts[kw1].append(kw2)
                                keyword_contexts[kw2].append(kw1)
            except json.JSONDecodeError:
                continue
    
    # 1. 统计高频关键词
    top_keywords = [kw for kw, _ in keyword_counter.most_common(50)]
    print(f"发现的高频关键词: {', '.join(top_keywords[:10])}...")
    
    # 2. 关键词聚类（发现主题）
    if len(top_keywords) < 2:
        print("关键词数量不足，无法进行聚类分析")
        return {"keyword_counter": keyword_counter, "top_keywords": top_keywords}
    
    vectorizer = CountVectorizer(vocabulary=top_keywords)
    keyword_vectors = []
    for kw in top_keywords:
        context_str = ' '.join(keyword_contexts.get(kw, []))
        vec = vectorizer.transform([context_str]).toarray()[0]
        keyword_vectors.append(vec)
    
    X = np.array(keyword_vectors)
    clustering = DBSCAN(
        eps=config["clustering"]["eps"], 
        min_samples=config["clustering"]["min_samples"]
    ).fit(X)
    
    topics = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        if label != -1:  # 只保留非噪声点
            topics[f"主题_{label}"].append(top_keywords[i])
    
    # 如果没有有效主题，直接返回
    if not topics:
        print("未发现有效主题聚类结果")
        return {
            "total_keywords": len(keyword_counter),
            "top_keywords": top_keywords,
            "keyword_frequency": dict(keyword_counter),
            "message": "未发现有效主题聚类"
        }
    
    # 3. 分析主题间的关系（相似度）
    topic_vectors = []
    topic_names = []
    for topic_id, keywords in topics.items():
        # 过滤出在top_keywords中的有效关键词
        valid_keywords = [kw for kw in keywords if kw in top_keywords]
        if not valid_keywords:
            continue  # 跳过没有有效关键词的主题
        
        # 计算主题向量（关键词向量的平均值）
        topic_vec = np.mean(
            [keyword_vectors[top_keywords.index(kw)] for kw in valid_keywords],
            axis=0
        )
        topic_vectors.append(topic_vec)
        topic_names.append(topic_id)
    
    # 处理可能的空主题向量列表
    topic_similarity = {}
    if len(topic_vectors) >= 1:
        # 确保输入是2D数组
        topic_vectors_np = np.array(topic_vectors)
        if topic_vectors_np.ndim == 1:
            topic_vectors_np = topic_vectors_np.reshape(-1, 1)
        
        try:
            topic_similarity_matrix = cosine_similarity(topic_vectors_np)
            # 转换为字典格式
            topic_similarity = {
                topic_names[i]: {
                    topic_names[j]: float(topic_similarity_matrix[i][j]) 
                    for j in range(len(topic_names)) if i != j
                } 
                for i in range(len(topic_names))
            }
        except Exception as e:
            print(f"计算主题相似度时出错: {str(e)}")
            topic_similarity = {}
    else:
        print("没有足够的有效主题向量用于计算相似度")
    
    # 4. 为主题生成有意义的名称
    topic_naming = {}
    for topic_id, keywords in topics.items():
        category_scores = defaultdict(int)
        for kw in keywords:
            for category, terms in config["financial_dict"].items():
                if kw in terms:
                    category_scores[category] += 1
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            topic_naming[topic_id] = best_category
        else:
            topic_naming[topic_id] = f"主题: {', '.join(keywords[:3])}"
    
    # 5. 构建主题-次主题关系
    final_topic_tree = defaultdict(lambda: defaultdict(list))
    
    for topic_id, keywords in topics.items():
        main_topic = topic_naming[topic_id]
        
        subtopics = defaultdict(list)
        for kw in keywords:
            matched = False
            for first_level, second_levels in config["topic_tree"].items():
                if main_topic in first_level:
                    for second_level in second_levels:
                        if any(term in second_level for term in kw.split()):
                            subtopics[second_level].append(kw)
                            matched = True
                            break
                    if matched:
                        break
            if not matched:
                subtopics["其他"].append(kw)
        
        for subtopic, terms in subtopics.items():
            final_topic_tree[main_topic][subtopic] = list(set(terms))
    
    # 保存分析结果
    output_path = os.path.join(config["paths"]["output_dir"], "final_topic_relations.json")
    result = {
        "total_keywords": len(keyword_counter),
        "top_keywords": top_keywords,
        "keyword_frequency": dict(keyword_counter),
        "topic_clusters": {k: v for k, v in topics.items()},
        "topic_names": topic_naming,
        "topic_similarity": topic_similarity,
        "topic_hierarchy": final_topic_tree,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"主题关系分析完成，结果保存至：{output_path}")
    return result


# --------------------------
# 第三阶段：自动更新配置
# --------------------------
def auto_update_config(config: Dict, analysis_result: Dict) -> None:
    """自动更新financial_dict、topic_tree和stopwords"""
    print("\n开始自动更新配置...")
    
    # 确保更新目录存在
    update_dir = config["paths"]["update_dir"]
    os.makedirs(update_dir, exist_ok=True)
    
    # 保存原始配置作为备份
    backup_path = os.path.join(update_dir, f"config_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"配置备份已保存至：{backup_path}")
    
    # 1. 更新financial_dict
    update_params = config["auto_update"]
    keyword_freq = analysis_result.get("keyword_frequency", {})
    current_terms = set([term for terms in config["financial_dict"].values() for term in terms])
    
    # 发现新术语：高频且不在现有词典中
    new_terms = []
    for kw, freq in keyword_freq.items():
        if (freq >= update_params["min_term_frequency"] and 
            kw not in current_terms and 
            len(kw) >= update_params["min_term_length"]):
            new_terms.append(kw)
    
    # 将新术语添加到最相关的类别
    if new_terms and update_params["update_financial_dict"]:
        print(f"发现 {len(new_terms)} 个潜在新术语，正在分类...")
        for term in new_terms:
            best_category = None
            max_similarity = 0
            
            # 找到最相关的类别
            for category, terms in config["financial_dict"].items():
                if not terms:
                    continue
                # 使用Ollama判断术语最可能属于哪个类别
                if config["model"]["type"] == "ollama":
                    try:
                        prompt = f"""以下术语最可能属于哪个金融类别？请只返回类别名称，不做解释。
术语：{term}
类别选项：{', '.join(config['financial_dict'].keys())}
"""
                        response = call_ollama_model(prompt, config["model"]["params"])
                        if response in config["financial_dict"]:
                            best_category = response
                    except:
                        pass
                
                # 如果模型无法判断，使用简单匹配
                if not best_category:
                    category_terms = ' '.join(terms)
                    similarity = fuzz.partial_ratio(term, category_terms)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_category = category
            
            if best_category and max_similarity > update_params["min_similarity_threshold"]:
                config["financial_dict"][best_category].append(term)
                print(f"添加新术语 '{term}' 到类别 '{best_category}'")
    
    # 2. 更新topic_tree
    if update_params["update_topic_tree"] and "topic_hierarchy" in analysis_result:
        topic_hierarchy = analysis_result["topic_hierarchy"]
        for main_topic, subtopics in topic_hierarchy.items():
            # 添加新的一级主题
            if main_topic not in config["topic_tree"]:
                config["topic_tree"][main_topic] = []
                print(f"添加新的一级主题: {main_topic}")
            
            # 添加新的二级主题
            for subtopic in subtopics:
                if subtopic not in config["topic_tree"][main_topic] and subtopic != "其他":
                    config["topic_tree"][main_topic].append(subtopic)
                    print(f"为主题 '{main_topic}' 添加次主题: {subtopic}")
    
    # 3. 更新stopwords
    if update_params["update_stopwords"] and "keyword_frequency" in analysis_result:
        keyword_freq = analysis_result["keyword_frequency"]
        # 识别高频但无实际意义的词（太短或在多个主题中出现）
        for kw, freq in keyword_freq.items():
            if (freq >= update_params["min_stopword_frequency"] and 
                len(kw) <= update_params["max_stopword_length"] and 
                kw not in config["stopwords"]):
                config["stopwords"].append(kw)
                print(f"添加新停用词: {kw}")
    
    # 保存更新后的配置
    updated_config_path = os.path.join(update_dir, "updated_config.json")
    with open(updated_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"更新后的配置已保存至：{updated_config_path}")
    print("自动更新配置完成")


# --------------------------
# 主函数
# --------------------------
def main() -> None:
    # 配置参数（明确化各种方法的参数）
    config = {
        "paths": {
            "json_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/output_image_processed/elements_jsons",
            "output_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/output_image_processed/contents_theme",
            "progress_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/extract_theme_from_content_progress.json",
            "temp_keywords_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/temp_keywords.jsonl",
            "update_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/config_updates"  # 新增：配置更新目录
        },
        "processing": {
            "batch_size": 5,
            "max_retries": 3,
            "min_text_length": 50,
            "delete_progress_after_complete": False,  # 建议保留进度用于后续更新
            "max_total_duration": 3600,
            "max_batch_duration": 600,
            "max_file_duration": 120
        },
        "model": {
            "type": "ollama",  # "ollama" / "api" / "traditional"
            "params": {
                # Ollama参数
                "host": "http://localhost:11434",
                "model_name": "llama3:latest",
                "stream": False,
                "timeout": 30,
                "temperature": 0.3,  # 控制随机性，0-1之间
                "top_p": 0.9,        # 核采样参数
                "top_k": 40,         # 采样候选词数量
                
                # API参数（与Ollama参数部分重叠，使用时会自动选择对应参数）
                "api_url": "https://api.siliconflow.cn/v1/chat/completions",
                "api_key": "",
                "max_tokens": 100,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        },
        "traditional_method": {
            "tfidf": {
                "max_features": 1000,  # 最大特征数
                "min_df": 1,           # 最小文档频率
                "max_df": 0.95         # 最大文档频率
            },
            "textrank": {
                "window": 2,           # 窗口大小
                "lower": True,         # 是否转为小写
                "word_min_len": 2,     # 关键词最小长度
                "min_text_length": 10  # 最小文本长度
            }
        },
        "topic_extraction": {
            "top_k_keywords": 7,    # 提取的关键词数量
            "textrank_window": 2
        },
        "clustering": {
            "eps": 0.5,             # DBSCAN聚类参数
            "min_samples": 2        # 形成簇的最小样本数
        },
        "auto_update": {
            "enable": True,         # 是否启用自动更新
            "update_financial_dict": True,  # 是否更新金融词典
            "update_topic_tree": True,      # 是否更新主题树
            "update_stopwords": True,       # 是否更新停用词
            "min_term_frequency": 10,       # 新术语最小出现频率
            "min_term_length": 2,           # 新术语最小长度
            "min_stopword_frequency": 50,   # 停用词最小出现频率
            "max_stopword_length": 1,       # 停用词最大长度
            "min_similarity_threshold": 50  # 最小相似度阈值
        },
        "financial_dict": {
            "金融产品": [
                "股票", "普通股", "优先股", "国债", "企业债", "可转债", 
                "公募基金", "私募基金", "ETF", "LOF", "期货", "期权", 
                "外汇", "黄金ETF", "REITs", "同业存单", "结构性存款"
            ],
            "财务指标": [
                "营业收入", "归母净利润", "毛利率", "净利率", "ROE（净资产收益率）", 
                "ROA（资产收益率）", "流动比率", "速动比率", "资产负债率", "应收账款周转率", 
                "市盈率（PE）", "市净率（PB）", "市销率（PS）", "EPS（每股收益）", "股息率", 
                "经营性现金流净额", "同比增长率", "环比增长率"
            ],
            "资本动作": [
                "IPO（首次公开募股）", "再融资", "定向增发", "配股", "可转债发行", 
                "股票回购", "分红派息", "并购重组", "借壳上市", "退市", 
                "资产剥离", "股权转让", "债转股", "跨境并购"
            ],
            "市场术语": [
                "牛市", "熊市", "震荡市", "成交量", "成交额", "换手率", "振幅", 
                "跌停板", "涨停板", "主力资金", "北向资金", "南向资金", 
                "基本面分析", "技术分析", "量化交易", "止损", "止盈"
            ],
            "监管与政策": [
                "央行降准", "央行加息", "LPR（贷款市场报价利率）", "存款准备金率", 
                "美联储加息", "巴塞尔协议", "资管新规", "IPO注册制", "退市新规", 
                "外汇管制", "反垄断审查", "穿透式监管", "合格投资者制度"
            ]
        },
        "topic_tree": {
            "金融产品与市场": [
                "产品特性", "市场流动性", "产品风险等级", "投资者适当性", 
                "跨市场产品联动", "衍生品定价"
            ],
            "财务与估值": [
                "盈利质量分析", "偿债能力评估", "估值合理性判断", "财务舞弊识别", 
                "现金流健康度", "业绩预告解读"
            ],
            "资本运作": [
                "IPO进程与估值", "再融资方案分析", "并购重组可行性", "股东增减持影响", 
                "回购对股价影响", "分红政策解读"
            ],
            "监管与政策影响": [
                "货币政策传导", "监管新规解读", "跨境政策协同", "行业监管收紧/放松影响", 
                "政策套利空间"
            ],
            "投资策略": [
                "价值投资", "成长投资", "趋势跟踪", "对冲策略", "资产配置模型", 
                "行业轮动策略", "事件驱动策略"
            ],
            "风险分析": [
                "市场风险", "信用风险", "流动性风险", "操作风险", "政策风险", 
                "汇率风险", "利率风险", "黑天鹅事件影响"
            ]
        },
        "stopwords": [
            "证", "券", "研", "究", "报", "告", "公", "司", "页", 
            "资料来源", "wind", "同花顺", "东方财富", "日期", "单位", 
            "亿元", "万元", "图表", "制表", "如下", "所示", "注", "注释", 
            "摘要", "关键词", "引言", "结论", "分析师", "联系人", "电话", 
            "邮箱", "数据", "统计", "截至", "报告期", "季度", "年度"
        ]
    }
    
    # 初始化
    init_jieba(config["financial_dict"])
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    os.makedirs(config["paths"]["update_dir"], exist_ok=True)
    
    # 检查模型配置
    if config["model"]["type"] == "ollama":
        print("检查Ollama服务状态...")
        if not check_ollama_status(config["model"]["params"]["host"]):
            print(f"Ollama服务未运行在 {config['model']['params']['host']}")
            choice = input("是否继续并使用传统方法提取关键词? (y/n): ").strip().lower()
            if choice != 'y':
                print("程序终止")
                return
            config["model"]["type"] = "traditional"
    elif config["model"]["type"] == "api":
        if not config["model"]["params"]["api_key"] or config["model"]["params"]["api_key"].startswith("your-api"):
            print("API密钥未设置，将使用传统方法")
            config["model"]["type"] = "traditional"
    
    # 获取所有JSON文件
    all_json_files = get_json_files(config["paths"]["json_dir"])
    if not all_json_files:
        print("未找到任何JSON文件，程序终止")
        return
    print(f"共发现 {len(all_json_files)} 个JSON文件")
    
    # 加载进度
    progress = load_progress(config["paths"]["progress_file"])
    
    # 更新总文件数
    progress["total_files"] = len(all_json_files)
    save_progress(config["paths"]["progress_file"], progress)
    
    # 显示进度摘要
    print_progress_summary(progress)
    
    # 第一阶段：提取所有主题词到临时文件
    if not progress.get("completed_extraction", False):
        print("\n===== 开始第一阶段：提取主题词 =====")
        
        if not os.path.exists(config["paths"]["temp_keywords_file"]):
            open(config["paths"]["temp_keywords_file"], 'w', encoding='utf-8').close()
        
        processed = set(progress["processed_files"])
        failed = set(progress["failed_files"])
        processing = set(progress["processing_files"])
        pending_files = [f for f in all_json_files if f not in processed and f not in failed and f not in processing]
        print(f"待处理文件：{len(pending_files)} 个")
        
        total_batches = (len(pending_files) + config["processing"]["batch_size"] - 1) // config["processing"]["batch_size"]
        batches = [
            pending_files[i:i+config["processing"]["batch_size"]] 
            for i in range(0, len(pending_files), config["processing"]["batch_size"])
        ]
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="主题词提取进度")):
            if time.time() - start_time > config["processing"]["max_total_duration"]:
                print(f"总处理时间超过{config['processing']['max_total_duration']}秒，终止")
                break
            
            print(f"\n处理批次 {batch_idx+1}/{total_batches}（{len(batch)}个文件）")
            process_batch_extraction(batch, config, progress)
            print_progress_summary(progress)
        
        progress["completed_extraction"] = True
        save_progress(config["paths"]["progress_file"], progress)
        print("\n===== 第一阶段：主题词提取完成 =====")
    
    # 第二阶段：分析主题关系
    analysis_result = {}
    if progress.get("completed_extraction", False) and not progress.get("completed_analysis", False):
        print("\n===== 开始第二阶段：分析主题关系 =====")
        analysis_result = analyze_topic_relations(config)
        
        progress["completed_analysis"] = True
        save_progress(config["paths"]["progress_file"], progress)
        print("===== 第二阶段：主题关系分析完成 =====")
    
    # 第三阶段：自动更新配置
    if (config["auto_update"]["enable"] and 
        progress.get("completed_analysis", False) and 
        not progress.get("completed_updating", False) and 
        analysis_result):
        print("\n===== 开始第三阶段：自动更新配置 =====")
        auto_update_config(config, analysis_result)
        
        progress["completed_updating"] = True
        save_progress(config["paths"]["progress_file"], progress)
        print("===== 第三阶段：自动更新配置完成 =====")
    
    # 打印最终进度
    print_progress_summary(progress)
    
    # 清理临时文件
    if (config["processing"]["delete_progress_after_complete"] and 
        progress.get("completed_updating", False)):
        delete_progress(config["paths"]["progress_file"])
        if os.path.exists(config["paths"]["temp_keywords_file"]):
            try:
                os.remove(config["paths"]["temp_keywords_file"])
                print(f"已删除临时关键词文件：{config['paths']['temp_keywords_file']}")
            except Exception as e:
                print(f"删除临时关键词文件失败：{str(e)}")

if __name__ == "__main__":
    main()
