import os
import json
import re
from collections import defaultdict, Counter
import numpy as np
import requests
import jieba

# 忽略jieba的pkg_resources警告
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")

# --------------------------
# 常量与配置
# --------------------------
# 停用词表
STOPWORDS = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}


# --------------------------
# Ollama 工具函数
# --------------------------
def check_ollama_connection(params: dict) -> bool:
    """检查Ollama服务是否可连接"""
    try:
        response = requests.get(f"{params['host']}/", timeout=params["timeout"])
        return response.status_code == 200
    except Exception as e:
        print(f"Ollama服务连接失败: {str(e)}")
        return False

def get_available_ollama_models(params: dict) -> list:
    """获取Ollama可用的模型列表"""
    try:
        response = requests.get(f"{params['host']}/api/tags", timeout=params["timeout"])
        if response.status_code == 200:
            return [model["name"] for model in response.json().get("models", [])]
        else:
            print(f"获取Ollama模型列表失败，状态码: {response.status_code}")
            return []
    except Exception as e:
        print(f"获取Ollama模型列表错误: {str(e)}")
        return []

def validate_ollama_embedding_model(params: dict) -> str:
    """验证并返回可用的Ollama嵌入模型"""
    available_models = get_available_ollama_models(params)
    if not available_models:
        print("未找到任何可用的Ollama模型，请确保已安装模型")
        return None
    
    # 检查配置的嵌入模型是否可用
    if params["embedding_model"] in available_models:
        return params["embedding_model"]
    
    # 如果配置的模型不可用，尝试推荐一个可用的嵌入模型
    embedding_candidates = [
        model for model in available_models 
        if any(keyword in model.lower() for keyword in ["embed", "embedding", "bge", "gte"])
    ]
    
    if embedding_candidates:
        print(f"配置的嵌入模型 {params['embedding_model']} 不可用，将使用 {embedding_candidates[0]} 替代")
        return embedding_candidates[0]
    elif available_models:
        # 如果没有找到专用嵌入模型，使用第一个可用模型
        print(f"未找到专用嵌入模型，将尝试使用 {available_models[0]} 作为嵌入模型")
        return available_models[0]
    else:
        return None


# --------------------------
# 文本预处理工具
# --------------------------
def preprocess_text(text: str) -> str:
    """预处理文本：清洗、分词、去停用词"""
    if not text or not isinstance(text, str):
        return ""
    
    # 清洗特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 分词并去停用词
    words = jieba.cut(text)
    return " ".join([w for w in words if w.strip() and w not in STOPWORDS])


# --------------------------
# 嵌入生成函数 - 根据类型调用不同实现
# --------------------------
def get_embedding(text: str, model_config: dict) -> np.ndarray:
    """根据配置获取文本的嵌入向量"""
    processed_text = preprocess_text(text)
    if not processed_text:
        return np.array([])
    
    # 根据模型类型调用不同的嵌入生成方法
    if model_config["type"] == "ollama":
        return call_ollama_embedding(processed_text, model_config["params"])
    elif model_config["type"] == "api":
        return call_api_embedding(processed_text, model_config["params"])
    else:
        raise ValueError(f"不支持的模型类型: {model_config['type']}")


# --------------------------
# Ollama 调用函数
# --------------------------
def call_ollama_embedding(text: str, params: dict) -> np.ndarray:
    """调用Ollama生成嵌入向量"""
    try:
        data = {
            "model": params["embedding_model"],
            "prompt": text
        }
        response = requests.post(
            f"{params['host']}/api/embeddings",
            json=data,
            timeout=params["timeout"]
        )
        
        # 处理404错误 - 模型未找到
        if response.status_code == 404:
            # 尝试获取可用模型并更新配置
            new_model = validate_ollama_embedding_model(params)
            if new_model:
                params["embedding_model"] = new_model
                print(f"已自动切换为可用的嵌入模型: {new_model}")
                # 用新模型重试
                data["model"] = new_model
                response = requests.post(
                    f"{params['host']}/api/embeddings",
                    json=data,
                    timeout=params["timeout"]
                )
            else:
                raise Exception(f"Ollama嵌入模型 {params['embedding_model']} 未找到，且无可用替代模型")
        
        if response.status_code == 200:
            return np.array(response.json().get("embedding", []))
        else:
            raise Exception(f"Ollama嵌入请求失败，状态码: {response.status_code}，响应内容: {response.text}")
    except Exception as e:
        raise Exception(f"Ollama嵌入调用错误: {str(e)}")

def call_ollama_model(prompt: str, params: dict) -> str:
    """调用Ollama生成文本"""
    try:
        data = {
            "model": params["model_name"],
            "prompt": prompt,
            "stream": params["stream"],
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
            raise Exception(f"Ollama请求失败，状态码: {response.status_code}，响应内容: {response.text}")
    except Exception as e:
        raise Exception(f"Ollama调用错误: {str(e)}")


# --------------------------
# API 调用函数
# --------------------------
def call_api_embedding(text: str, params: dict) -> np.ndarray:
    """调用API生成嵌入向量"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {params['api_key']}"
        }
        data = {
            "input": text
        }
        response = requests.post(
            params["embedding_api_url"],
            json=data,
            headers=headers,
            timeout=params["timeout"]
        )
        if response.status_code == 200:
            return np.array(response.json().get("data", [{}])[0].get("embedding", []))
        else:
            raise Exception(f"API嵌入请求失败，状态码: {response.status_code}")
    except Exception as e:
        raise Exception(f"API嵌入调用错误: {str(e)}")

def call_api_model(prompt: str, params: dict) -> str:
    """调用API生成文本"""
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
            "top_p": params["top_p"]
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
            raise Exception(f"API请求失败，状态码: {response.status_code}")
    except Exception as e:
        raise Exception(f"API调用错误: {str(e)}")


# --------------------------
# 检索工具函数 - 优化主题和内容处理逻辑
# --------------------------
def retrieve_relevant_topic_paths(query: str, topic_embeddings: dict, model_config: dict,
                                 top_n: int = 5, threshold: float = 0.5) -> list:
    """检索与查询最相关的主题路径（优化版）"""
    if not topic_embeddings or top_n <= 0:
        return []
    
    # 生成查询向量
    query_embedding = get_embedding(query, model_config)
    if len(query_embedding) == 0:
        return []
    
    # 计算与每个主题的相似度
    similarities = []
    for topic_path, topic_emb in topic_embeddings.items():
        try:
            # 计算余弦相似度
            similarity = np.dot(query_embedding, topic_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(topic_emb)
            ) if (np.linalg.norm(query_embedding) > 0 and np.linalg.norm(topic_emb) > 0) else 0
            
            if similarity >= threshold:
                similarities.append((topic_path, similarity))
        except Exception as e:
            print(f"计算主题 {topic_path} 相似度失败: {str(e)}")
            continue
    
    # 按相似度排序并取前n个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def get_content_ids_from_topics(topic_paths: list, topic_data: dict, top_k: int = 10) -> list:
    """从相关主题中获取内容ID（优化版，适配新的topics结构）"""
    if not topic_paths or not topic_data:
        return []
    
    content_scores = defaultdict(float)
    
    # 为每个主题下的内容评分（结合主题相似度和内容相似度）
    for topic_path, topic_similarity in topic_paths:
        if topic_path not in topic_data:
            continue
            
        # 从主题数据中获取内容列表（适配新结构中的"contents"字段）
        topic_contents = topic_data[topic_path].get("contents", [])
        for content in topic_contents:
            # 从内容项中提取ID和相似度（适配新结构中的"id"和"similarity"字段）
            content_id = content.get("id")
            content_similarity = content.get("similarity", 0)
            
            if content_id:
                # 综合评分 = 主题相似度 * 内容相似度
                combined_score = topic_similarity * content_similarity
                content_scores[content_id] += combined_score
    
    # 按综合评分排序并取前k个
    sorted_contents = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_contents[:top_k]]

def refine_content_retrieval(query: str, content_ids: list, all_contents: dict, model_config: dict,
                            top_n: int = 3, threshold: float = 0.5) -> list:
    """对检索到的内容ID进行二次筛选（优化版，适配新的all_contents结构）"""
    if not content_ids or not all_contents or top_n <= 0:
        return []
    
    # 生成查询向量
    query_embedding = get_embedding(query, model_config)
    if len(query_embedding) == 0:
        return []
    
    # 获取内容并计算相似度
    content_similarities = []
    for content_id in content_ids:
        # 从all_contents中获取内容详情
        content = all_contents.get(content_id)
        if not content:
            continue
            
        # 从内容中提取文本（适配新结构中的"content"字段）
        content_text = content.get("content", "")
        if not content_text:
            continue
            
        # 生成内容嵌入
        content_embedding = get_embedding(content_text, model_config)
        if len(content_embedding) == 0:
            continue
            
        # 计算余弦相似度
        similarity = np.dot(query_embedding, content_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
        ) if (np.linalg.norm(query_embedding) > 0 and np.linalg.norm(content_embedding) > 0) else 0
        
        if similarity >= threshold:
            content_similarities.append((content, similarity))
    
    # 按相似度排序并取前n个
    content_similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in content_similarities[:top_n]]


# --------------------------
# 答案生成函数
# --------------------------
def generate_answer(question: str, contexts: list, model_config: dict) -> str:
    """使用检索到的上下文生成答案"""
    if not contexts:
        return json.dumps({
            "answer": "没有找到相关信息", 
            "retrieved_content": "", 
            "filename": "", 
            "page": 0
        }, ensure_ascii=False)
    
    # 构建上下文（适配新结构中的"content"字段）
    context_text = "\n\n".join([f"[{i+1}] {ctx.get('content', '')}" for i, ctx in enumerate(contexts)])
    
    # 构建提示词
    prompt = f"""请根据以下检索到的内容，简洁准确地回答问题。
检索内容：
{context_text}

问题：{question}

请严格按照如下JSON格式输出：
{{"answer": "你的简洁回答", "retrieved_content": "{context_text}", "filename": "", "page": 0}}
请确保输出内容为合法JSON字符串，不要输出多余内容。"""
    
    # 根据模型类型调用不同的生成方法
    try:
        if model_config["type"] == "ollama":
            response = call_ollama_model(prompt, model_config["params"])
        elif model_config["type"] == "api":
            response = call_api_model(prompt, model_config["params"])
        else:
            # 默认返回基于规则的简单答案
            response = json.dumps({
                "answer": f"根据提供的信息，关于'{question}'的内容如下：{context_text[:200]}...",
                "retrieved_content": context_text,
                "filename": "",
                "page": 0
            }, ensure_ascii=False)
        
        # 提取最常见的filename和page（适配新结构中的"filename"和"page"字段）
        filenames = [ctx.get("filename", "") for ctx in contexts if ctx.get("filename")]
        pages = [ctx.get("page", 0) for ctx in contexts if ctx.get("page")]
        
        most_common_filename = filenames[0] if filenames else ""
        most_common_page = pages[0] if pages else 0
        
        if filenames:
            most_common_filename = Counter(filenames).most_common(1)[0][0]
        if pages:
            most_common_page = Counter(pages).most_common(1)[0][0]
        
        # 更新答案中的filename和page
        try:
            answer_json = json.loads(response)
            answer_json["retrieved_content"] = context_text
            answer_json["filename"] = most_common_filename
            answer_json["page"] = most_common_page
            return json.dumps(answer_json, ensure_ascii=False)
        except:
            return json.dumps({
                "answer": response,
                "retrieved_content": context_text,
                "filename": most_common_filename,
                "page": most_common_page
            }, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({
            "answer": f"生成答案时出错: {str(e)}",
            "retrieved_content": context_text,
            "filename": "",
            "page": 0
        }, ensure_ascii=False)


# --------------------------
# 数据加载函数
# --------------------------
def load_topic_mapping_data(file_path: str) -> dict:
    """加载主题映射JSON数据（优化版，适配新结构）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"加载主题映射数据失败: {str(e)}")
        return {}


# --------------------------
# 主函数
# --------------------------
def main():
    # --------------------------
    # 参数配置 - 所有参数在此处设置
    # --------------------------
    params = {
        "topic_mapping_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_embeded_contents/topic_mappings_result.json",
        "questions_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/datas/test.json",
        "output_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/datas/predicted.json",
        
        # 检索参数
        "top_n_topics": 5,        # 检索的主题数量
        "top_k_content_ids": 20,  # 获取的内容ID数量
        "top_n_contents": 3,      # 最终选择的内容数量
        
        # 相似度阈值
        "topic_similarity_threshold": 0.4,    # 主题相似度阈值
        "content_similarity_threshold": 0.5,  # 内容相似度阈值
        
        # 生成模型参数 - 通过type指定调用方式
        "model": {
            "type": "ollama",  # 可选: "ollama", "api"
            "params": {
                # Ollama参数
                "host": "http://localhost:11434",
                "model_name": "llama3:latest",
                "embedding_model": "bge-large-en-v1.5",  # Ollama可用的嵌入模型
                "timeout": 60,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "stream": False,
                
                # API参数 (当type为"api"时使用)
                "api_url": "https://api.siliconflow.cn/v1/chat/completions",
                "embedding_api_url": "https://api.siliconflow.cn/v1/embeddings",
                "api_key": "",
                "max_tokens": 500
            }
        }
    }
    
    # --------------------------
    # Ollama 连接与模型验证
    # --------------------------
    if params["model"]["type"] == "ollama":
        print("验证Ollama连接和模型...")
        # 检查Ollama服务是否可用
        if not check_ollama_connection(params["model"]["params"]):
            print("无法连接到Ollama服务，请确保Ollama已启动并正在运行")
            return
        
        # 验证并更新嵌入模型
        valid_embedding_model = validate_ollama_embedding_model(params["model"]["params"])
        if not valid_embedding_model:
            print("没有可用的Ollama嵌入模型，请先使用 'ollama pull' 命令安装一个嵌入模型")
            print("推荐安装: ollama pull bge-large-en-v1.5 或 ollama pull mxbai-embed-large")
            return
        params["model"]["params"]["embedding_model"] = valid_embedding_model
        
        # 验证生成模型
        available_models = get_available_ollama_models(params["model"]["params"])
        if params["model"]["params"]["model_name"] not in available_models:
            print(f"配置的生成模型 {params['model']['params']['model_name']} 不可用")
            if available_models:
                print(f"将使用 {available_models[0]} 作为生成模型")
                params["model"]["params"]["model_name"] = available_models[0]
            else:
                print("没有可用的生成模型，请先安装一个生成模型")
                return
    
    # --------------------------
    # 加载数据并适配新结构
    # --------------------------
    print("加载数据...")
    topic_mapping_data = load_topic_mapping_data(params["topic_mapping_file"])
    
    # 提取主题映射数据中的关键部分（适配新结构）
    topics = topic_mapping_data.get("topics", {})  # 主题数据
    all_contents = topic_mapping_data.get("all_contents", {})  # 所有内容数据
    uncategorized = topic_mapping_data.get("uncategorized", [])  # 未分类内容
    failed = topic_mapping_data.get("failed", [])  # 处理失败内容
    
    # 打印数据统计信息
    print(f"加载完成 - 主题数: {len(topics)}, 内容数: {len(all_contents)}, "
          f"未分类: {len(uncategorized)}, 处理失败: {len(failed)}")
    
    # 提取主题嵌入（优化版，适配新结构）
    topic_embeddings = {}
    for topic_path, topic_data in topics.items():
        # 检查是否存在embedding键（新结构中可能直接存储在主题数据中）
        if "embedding" in topic_data:
            topic_embeddings[topic_path] = topic_data["embedding"]
        else:
            print(f"主题 {topic_path} 缺少嵌入数据，尝试生成...")
            # 从主题info中获取文本生成嵌入（适配新结构中的"info"字段）
            topic_info = topic_data.get("info", {})
            topic_text = topic_info.get("name", "")  # 使用主题名称作为文本
            
            # 如果主题名称为空，尝试使用主题路径作为文本
            if not topic_text:
                topic_text = topic_path
            
            try:
                embedding = get_embedding(topic_text, params["model"])
                if len(embedding) > 0:
                    topic_embeddings[topic_path] = embedding
                    print(f"成功为主题 {topic_path} 生成嵌入")
                else:
                    print(f"无法为主题 {topic_path} 生成嵌入，将跳过该主题")
            except Exception as e:
                print(f"生成主题 {topic_path} 嵌入失败: {str(e)}")
    
    # 检查是否有可用的主题嵌入
    if not topic_embeddings:
        print("警告：没有可用的主题嵌入数据，将尝试直接检索内容")
    
    # 加载问题
    try:
        with open(params["questions_file"], 'r', encoding='utf-8') as f:
            questions = json.load(f)
        if not isinstance(questions, list):
            questions = [questions]
    except Exception as e:
        print(f"加载问题文件失败: {str(e)}")
        questions = []
    
    if not questions:
        print("没有找到问题数据，程序退出")
        return
    
    # --------------------------
    # 处理每个问题
    # --------------------------
    answers = []
    print(f"开始处理 {len(questions)} 个问题...")
    
    for i, item in enumerate(questions):
        question = item.get("question", "")
        if not question:
            print(f"跳过第 {i+1} 个问题：问题为空")
            continue
        
        print(f"处理第 {i+1} 个问题：{question[:50]}...")
        
        # 1. 检索相关主题路径
        relevant_topics = []
        if topic_embeddings:  # 只有当有可用的主题嵌入时才进行主题检索
            relevant_topics = retrieve_relevant_topic_paths(
                question, 
                topic_embeddings,
                params["model"],  # 传递模型配置
                params["top_n_topics"],
                params["topic_similarity_threshold"]
            )
        
        # 如果没有找到相关主题，尝试直接从所有内容中检索
        if not relevant_topics:
            print(f"第 {i+1} 个问题没有找到相关主题，尝试直接检索内容...")
            # 直接使用所有内容ID
            content_ids = list(all_contents.keys())[:params["top_k_content_ids"]]
        else:
            # 2. 从相关主题获取内容ID
            content_ids = get_content_ids_from_topics(
                relevant_topics,
                topics,
                params["top_k_content_ids"]
            )
        
        if not content_ids:
            print(f"第 {i+1} 个问题没有找到相关内容ID")
            answer = {
                "question": question,
                "answer": "没有找到相关内容",
                "retrieved_content": "",
                "filename": "",
                "page": 0
            }
            answers.append(answer)
            continue
        
        # 3. 二次筛选内容
        selected_contents = refine_content_retrieval(
            question,
            content_ids,
            all_contents,
            params["model"],  # 传递模型配置
            params["top_n_contents"],
            params["content_similarity_threshold"]
        )
        
        if not selected_contents:
            print(f"第 {i+1} 个问题没有找到足够相关的内容")
            answer = {
                "question": question,
                "answer": "没有找到足够相关的内容",
                "retrieved_content": "",
                "filename": "",
                "page": 0
            }
            answers.append(answer)
            continue
        
        # 4. 生成答案
        answer_json = generate_answer(question, selected_contents, params["model"])
        try:
            answer_data = json.loads(answer_json)
            answer_data["question"] = question
            answers.append(answer_data)
            print(f"第 {i+1} 个问题处理完成")
        except Exception as e:
            print(f"解析答案JSON失败: {str(e)}")
            answers.append({
                "question": question,
                "answer": "生成答案格式错误",
                "retrieved_content": "",
                "filename": "",
                "page": 0
            })
    
    # 保存所有答案
    with open(params["output_file"], 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    print(f"所有问题处理完成，结果已保存至 {params['output_file']}")


if __name__ == "__main__":
    main()
