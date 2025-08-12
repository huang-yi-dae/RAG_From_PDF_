import re
import time
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import jieba
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# 日志配置 - 从配置中读取参数
# --------------------------
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """根据配置设置日志"""
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO").upper()
    log_file = log_config.get("file", "topic_mapping.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_config.get("format", '%(asctime)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 先初始化基础日志，配置加载后会重新设置
logger = logging.getLogger(__name__)

# --------------------------
# 常量定义
# --------------------------
PROCESS_STATUS = {
    "UNPROCESSED": "未处理",
    "PROCESSING": "处理中",
    "COMPLETED": "已完成",
    "FAILED": "处理失败",
    "UNCATEGORIZED": "未分类"
}

# --------------------------
# 模型处理基类
# --------------------------
class BaseModelProcessor:
    """模型处理器基类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.params = config.get("params", {})
        self._initialize()
    
    def _initialize(self) -> None:
        """初始化模型"""
        raise NotImplementedError("子类必须实现初始化方法")
    
    def get_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        raise NotImplementedError("子类必须实现关键词提取方法")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入"""
        raise NotImplementedError("子类必须实现嵌入生成方法")

# --------------------------
# Ollama模型处理器
# --------------------------
class OllamaModelProcessor(BaseModelProcessor):
    """Ollama模型处理器"""
    def _initialize(self) -> None:
        self.host = self.params.get("host", "http://localhost:11434")
        self.model_name = self.params.get("model_name", "llama3:latest")
        self.timeout = self.params.get("timeout", 30)
        
        # 测试连接
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if any(self.model_name in name for name in model_names):
                    logger.info(f"成功连接到Ollama，模型 {self.model_name} 可用")
                else:
                    logger.warning(f"Ollama中未找到 {self.model_name} 模型，可能需要先拉取")
            else:
                logger.warning(f"连接Ollama失败，状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama初始化失败: {str(e)}")
    
    def get_keywords(self, text: str) -> List[str]:
        """使用Ollama提取关键词"""
        prompt = f"""请从以下文本中提取最相关的{self.params.get('top_k', 5)}个关键词，
        仅返回关键词列表，用逗号分隔，不要其他内容：\n{text[:500]}"""
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": self.params.get("stream", False),
                    "temperature": self.params.get("temperature", 0.3),
                    "top_p": self.params.get("top_p", 0.9),
                    "top_k": self.params.get("top_k", 40)
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                return [kw.strip() for kw in content.split(",") if kw.strip()]
            else:
                logger.error(f"Ollama请求失败，状态码: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Ollama提取关键词失败: {str(e)}")
            return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """使用Ollama生成嵌入"""
        try:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text[:1000]  # 限制文本长度
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                return np.array(embedding) if embedding else np.array([])
            else:
                logger.error(f"Ollama嵌入请求失败，状态码: {response.status_code}")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Ollama生成嵌入失败: {str(e)}")
            return np.array([])

# --------------------------
# API模型处理器
# --------------------------
class ApiModelProcessor(BaseModelProcessor):
    """API模型处理器（兼容OpenAI格式API）"""
    def _initialize(self) -> None:
        self.api_url = self.params.get("api_url", "https://api.siliconflow.cn/v1/chat/completions")
        self.api_key = self.params.get("api_key", "")
        
        if not self.api_key:
            logger.warning("API密钥未配置，可能导致请求失败")
        else:
            logger.info(f"API模型处理器初始化完成，使用接口: {self.api_url}")
    
    def get_keywords(self, text: str) -> List[str]:
        """通过API提取关键词"""
        if not self.api_key:
            logger.error("API密钥未配置，无法提取关键词")
            return []
            
        prompt = f"""请从以下文本中提取最相关的{self.params.get('top_k', 5)}个关键词，
        仅返回关键词列表，用逗号分隔，不要其他内容：\n{text[:500]}"""
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.params.get("model_name", "gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.params.get("temperature", 0.3),
                    "max_tokens": self.params.get("max_tokens", 100),
                    "frequency_penalty": self.params.get("frequency_penalty", 0.0),
                    "presence_penalty": self.params.get("presence_penalty", 0.0)
                },
                timeout=self.params.get("timeout", 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return [kw.strip() for kw in content.split(",") if kw.strip()]
            else:
                logger.error(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"API提取关键词失败: {str(e)}")
            return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """通过API生成嵌入"""
        if not self.api_key:
            logger.error("API密钥未配置，无法生成嵌入")
            return np.array([])
            
        try:
            # 假设使用单独的嵌入API，根据实际情况调整
            embedding_url = self.params.get("embedding_url", "https://api.siliconflow.cn/v1/embeddings")
            response = requests.post(
                embedding_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.params.get("embedding_model", "text-embedding-ada-002"),
                    "input": text[:1000]
                },
                timeout=self.params.get("timeout", 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("data", [{}])[0].get("embedding", [])
                return np.array(embedding) if embedding else np.array([])
            else:
                logger.error(f"API嵌入请求失败，状态码: {response.status_code}, 响应: {response.text}")
                return np.array([])
                
        except Exception as e:
            logger.error(f"API生成嵌入失败: {str(e)}")
            return np.array([])

# --------------------------
# 传统方法处理器
# --------------------------
class TraditionalProcessor(BaseModelProcessor):
    """传统方法处理器（TF-IDF+TextRank）"""
    def _initialize(self) -> None:
        self.tfidf_config = self.config.get("traditional_method", {}).get("tfidf", {})
        self.textrank_config = self.config.get("traditional_method", {}).get("textrank", {})
        self.stopwords = self.config.get("stopwords", [])
        self.stopwords_set = set(self.stopwords)
        logger.info("传统方法处理器初始化完成")
    
    def get_keywords(self, text: str) -> List[str]:
        """使用TF-IDF提取关键词"""
        try:
            # 文本预处理
            text = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 分词并去停用词
            words = jieba.cut(text)
            processed_text = " ".join([w for w in words if w.strip() and w not in self.stopwords_set])
            
            if not processed_text:
                return []
            
            # 初始化TF-IDF向量器
            vectorizer = TfidfVectorizer(
                max_features=self.tfidf_config.get("max_features", 1000),
                min_df=self.tfidf_config.get("min_df", 1),
                max_df=self.tfidf_config.get("max_df", 0.95)
            )
            
            # 提取关键词
            X = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            weights = X.toarray()[0]
            sorted_indices = weights.argsort()[::-1]
            
            # 返回权重最高的关键词
            top_k = self.config.get("topic_extraction", {}).get("top_k_keywords", 7)
            return [feature_names[i] for i in sorted_indices if weights[i] > 0][:top_k]
            
        except Exception as e:
            logger.error(f"传统方法提取关键词失败: {str(e)}")
            return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """传统方法不生成嵌入，返回空数组"""
        logger.warning("传统方法不支持生成嵌入向量")
        return np.array([])

# --------------------------
# 模型工厂 - 根据配置创建相应模型处理器
# --------------------------
class ModelFactory:
    """模型工厂类，根据配置创建相应的模型处理器"""
    @staticmethod
    def create_processor(config: Dict[str, Any]) -> BaseModelProcessor:
        model_type = config.get("type", "traditional").lower()
        
        if model_type == "ollama":
            return OllamaModelProcessor(config)
        elif model_type == "api":
            return ApiModelProcessor(config)
        elif model_type == "traditional":
            return TraditionalProcessor(config)
        else:
            logger.warning(f"未知模型类型: {model_type}，将使用传统方法")
            return TraditionalProcessor(config)

# --------------------------
# 文件加载工具
# --------------------------
class FileLoader:
    """文件加载工具类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config.get("paths", {})
        self.encoding = config.get("processing", {}).get("encoding", "utf-8")
        
        # 确保输出目录存在
        for path_key in ["output_dir", "update_dir"]:
            if path_key in self.paths:
                Path(self.paths[path_key]).mkdir(parents=True, exist_ok=True)
    
    def load_stopwords(self) -> List[str]:
        """加载停用词"""
        stopwords = self.config.get("stopwords", [])
        stopwords_file = self.paths.get("stopwords_file")
        
        if not stopwords_file:
            return stopwords
            
        file_path = Path(stopwords_file)
        if file_path.exists() and file_path.is_file():
            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            stopwords.append(word)
                logger.info(f"从 {file_path} 加载了 {len(stopwords) - len(self.config.get('stopwords', []))} 个额外停用词")
            except Exception as e:
                logger.warning(f"加载停用词文件失败: {str(e)}，将使用默认停用词")
        
        return list(set(stopwords))  # 去重
    
    def load_processing_data(self) -> List[Dict[str, Any]]:
        """加载待处理的JSON数据"""
        input_dir = self.paths.get("json_dir")
        if not input_dir:
            raise ValueError("配置中未指定json_dir路径")
            
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            raise NotADirectoryError(f"输入目录不存在或不是有效的目录: {input_path}")
        
        processed_data = []
        json_files = list(input_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"在目录 {input_path} 中未找到任何JSON文件")
        
        logger.info(f"发现 {len(json_files)} 个JSON文件，开始加载...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding=self.encoding) as f:
                    data = json.load(f)
                
                file_basename = json_file.name
                for item in data:
                    if not isinstance(item, dict):
                        continue
                        
                    original_id = item.get("id", f"auto_{len(processed_data)}")
                    content_id = f"{file_basename}::{original_id}"
                    
                    # 从metadata提取page和filename
                    metadata = item.get("metadata", {})
                    page = metadata.get("page", "unknown")
                    filename = metadata.get("file_name", file_basename)
                    
                    processed_data.append({
                        "id": content_id,
                        "content": item.get("content", ""),
                        "page": page,  # 使用metadata中的page
                        "filename": filename,  # 使用metadata中的file_name
                        "source_path": str(json_file),
                        "original_metadata": metadata  # 保留完整metadata供后续使用
                    })
            
            except json.JSONDecodeError:
                logger.warning(f"文件 {json_file} 不是有效的JSON格式，已跳过")
                continue
            except Exception as e:
                logger.warning(f"加载文件 {json_file} 时出错: {str(e)}，已跳过")
                continue
        
        logger.info(f"成功加载所有JSON文件，共 {len(processed_data)} 条记录")
        return processed_data
    
    def load_topic_hierarchy(self) -> Dict[str, Any]:
        """加载主题层级结构"""
        hierarchy_file = self.paths.get("hierarchy_file")
        if not hierarchy_file:
            logger.warning("未配置主题层级文件，将使用默认主题结构")
            return {"topics": {}, "embeddings": {}, "topic_info": {}}
            
        file_path = Path(hierarchy_file)
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"主题层级文件不存在: {file_path}，将使用默认主题结构")
            return {"topics": {}, "embeddings": {}, "topic_info": {}}
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                hierarchy_data = json.load(f)
            
            topic_tree = hierarchy_data.get("topic_tree", {})
            financial_dict = hierarchy_data.get("financial_dict", {})
            
            if not topic_tree:
                raise ValueError("主题层级文件必须包含'topic_tree'字段")
            
            return {
                "topics": self._convert_topic_structure(topic_tree),
                "financial_dict": financial_dict,
                "embeddings": {},
                "topic_info": {}
            }
            
        except json.JSONDecodeError:
            raise ValueError(f"主题层级文件 {file_path} 不是有效的JSON格式")
        except Exception as e:
            raise ValueError(f"加载主题层级文件失败: {str(e)}")
    
    def _convert_topic_structure(self, topic_struct: Any) -> Dict[str, Any]:
        """转换主题结构为统一的字典格式"""
        if isinstance(topic_struct, list):
            return {item: {} for item in topic_struct}
        elif isinstance(topic_struct, dict):
            return {k: self._convert_topic_structure(v) for k, v in topic_struct.items()}
        else:
            return {}
    
    def save_progress(self, progress: Dict[str, Any]) -> None:
        """保存处理进度"""
        progress_file = self.paths.get("progress_file")
        if not progress_file:
            logger.warning("未配置进度文件路径，无法保存进度")
            return
            
        try:
            with open(progress_file, 'w', encoding=self.encoding) as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            logger.info(f"进度已保存至: {progress_file}")
        except Exception as e:
            logger.error(f"保存进度失败: {str(e)}")
    
    def load_progress(self) -> Dict[str, Any]:
        """加载已保存的进度"""
        progress_file = self.paths.get("progress_file")
        if not progress_file or not Path(progress_file).exists():
            return {"completed": [], "failed": [], "last_processed": None}
            
        try:
            with open(progress_file, 'r', encoding=self.encoding) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载进度失败: {str(e)}")
            return {"completed": [], "failed": [], "last_processed": None}
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """保存处理结果"""
        output_dir = self.paths.get("output_dir")
        if not output_dir:
            raise ValueError("配置中未指定output_dir路径")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result_filename = self.config.get("result_filename", "topic_mappings_result.json")
        result_file = output_path / result_filename
        
        try:
            with open(result_file, 'w', encoding=self.encoding) as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存至: {result_file}")
            return result_file
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise

# --------------------------
# 主题映射处理器
# --------------------------
class TopicMappingProcessor:
    """主题映射处理器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_config = config.get("processing", {})
        self.model_config = config.get("model", {})
        self.topic_extraction_config = config.get("topic_extraction", {})
        
        # 创建模型处理器
        self.processor = ModelFactory.create_processor(self.model_config)
        
        # 创建文件加载器
        self.file_loader = FileLoader(config)
        
        # 加载停用词
        self.stopwords = self.file_loader.load_stopwords()
        
        # 加载主题层级
        self.topic_hierarchy = self.file_loader.load_topic_hierarchy()
        
        # 生成主题嵌入（如果模型支持）
        self._generate_topic_embeddings()
    
    def _generate_topic_embeddings(self) -> None:
        """生成主题嵌入向量"""
        # 对于传统方法，不生成嵌入
        if self.model_config.get("type", "traditional").lower() == "traditional":
            return
            
        logger.info("开始生成主题嵌入向量...")
        topics = self.topic_hierarchy.get("topics", {})
        financial_dict = self.topic_hierarchy.get("financial_dict", {})
        
        def process_topic(topic_dict: Dict[str, Any], parent_path: str = "") -> None:
            for topic, subtopics in topic_dict.items():
                topic_full_path = f"{parent_path}.{topic}" if parent_path else topic
                
                # 构建主题文本
                topic_text = topic
                if isinstance(subtopics, dict) and subtopics:
                    topic_text += " " + " ".join(subtopics.keys())
                for category, terms in financial_dict.items():
                    if topic in terms or topic == category:
                        topic_text += " " + " ".join(terms)
                
                # 生成嵌入
                embedding = self.processor.get_embedding(topic_text)
                if embedding.size > 0:
                    self.topic_hierarchy["embeddings"][topic_full_path] = embedding
                    self.topic_hierarchy["topic_info"][topic_full_path] = {
                        "name": topic,
                        "parent_path": parent_path,
                        "levels": topic_full_path.split('.'),
                        "has_children": isinstance(subtopics, dict) and len(subtopics) > 0
                    }
                
                # 处理子主题
                if isinstance(subtopics, dict):
                    process_topic(subtopics, topic_full_path)
        
        process_topic(topics)
        logger.info(f"完成主题嵌入生成，共生成 {len(self.topic_hierarchy['embeddings'])} 个主题嵌入")
    
    def process_batch(self, data_batch: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """批处理数据"""
        results = {
            "topics": defaultdict(lambda: {"contents": [], "info": {}}),
            "all_contents": {},
            "uncategorized": [],
            "failed": []
        }
        
        # 初始化主题结构
        for topic_path, info in self.topic_hierarchy["topic_info"].items():
            results["topics"][topic_path]["info"] = info
        
        # 处理每个项目
        for item in tqdm(data_batch, desc="处理内容"):
            try:
                self._process_single_item(item, results)
            except Exception as e:
                content_id = item.get("id", f"unknown_{len(results['failed'])}")
                logger.error(f"处理内容 {content_id} 失败: {str(e)}")
                results["failed"].append({
                    "id": content_id,
                    "filename": item.get("filename", "unknown"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results, {
            "completed": [item["id"] for item in data_batch if item["id"] not in 
                         [f["id"] for f in results["failed"]]],
            "failed": [f["id"] for f in results["failed"]],
            "last_processed": datetime.now().isoformat()
        }
    
    def _process_single_item(self, item: Dict[str, Any], results: Dict[str, Any]) -> None:
        """处理单个内容项"""
        content_id = item["id"]
        content_text = item.get("content", "")
        
        # 检查内容长度
        min_length = self.processing_config.get("min_text_length", 50)
        if len(content_text.strip()) < min_length:
            results["uncategorized"].append({
                "id": content_id,
                "reason": f"内容过短（低于{min_length}字符阈值）",
                "page": item.get("page", "unknown"),
                "filename": item.get("filename", "unknown"),
                "content": content_text  # 保留完整内容
            })
            return
        
        # 记录内容基本信息 - 使用从metadata提取的page和filename
        results["all_contents"][content_id] = {
            "filename": item["filename"],
            "source_path": item["source_path"],
            "page": item["page"],
            "content": content_text,  # 保存完整内容
            "timestamp": datetime.now().isoformat(),
            "original_metadata": item.get("original_metadata", {})  # 保留原始metadata
        }
        
        # 提取关键词
        keywords = self.processor.get_keywords(content_text)
        if keywords:
            results["all_contents"][content_id]["keywords"] = keywords
        
        # 生成内容嵌入
        content_embedding = self.processor.get_embedding(content_text)
        if content_embedding.size > 0:
            results["all_contents"][content_id]["embedding"] = content_embedding.tolist()
        
        # 生成主题映射
        mappings = self._generate_mappings(content_text, content_embedding, keywords)
        
        # 记录映射结果
        for mapping in mappings:
            topic_path = mapping["topic_path"]
            if topic_path in results["topics"]:
                results["topics"][topic_path]["contents"].append({
                    "id": content_id,
                    "similarity": mapping["similarity"],
                    "method": mapping["method"]
                })
    
    def _generate_mappings(self, text: str, embedding: np.ndarray, keywords: List[str]) -> List[Dict[str, Any]]:
        """生成主题映射"""
        model_type = self.model_config.get("type", "traditional").lower()
        top_k = self.topic_extraction_config.get("top_k_keywords", 7)
        
        # 嵌入方法（Ollama或API）
        if model_type in ["ollama", "api"] and embedding.size > 0:
            return self._generate_embedding_mappings(embedding, top_k)
        
        # 传统方法（关键词匹配）
        return self._generate_traditional_mappings(text, keywords, top_k)
    
    def _generate_embedding_mappings(self, content_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """使用嵌入向量生成映射"""
        topic_embeddings = self.topic_hierarchy.get("embeddings", {})
        if not topic_embeddings:
            return []
        
        min_sim = self.model_config.get("min_embedding_similarity", 0.5)
        similarities = []
        
        for topic_path, topic_emb in topic_embeddings.items():
            try:
                sim_score = cosine_similarity([content_embedding], [topic_emb])[0][0]
                if sim_score >= min_sim:
                    similarities.append({
                        "topic_path": topic_path,
                        "similarity": float(sim_score),
                        "method": "embedding"
                    })
            except Exception as e:
                logger.warning(f"计算 {topic_path} 相似度失败: {str(e)}")
                continue
        
        # 按相似度排序并取前k个
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:top_k]
    
    def _generate_traditional_mappings(self, text: str, keywords: List[str], top_k: int) -> List[Dict[str, Any]]:
        """使用传统方法生成映射"""
        if not keywords:
            keywords = self.processor.get_keywords(text)
        if not keywords:
            return []
        
        match_results = defaultdict(int)
        min_threshold = self.config.get("auto_update", {}).get("min_similarity_threshold", 50)
        
        # 递归匹配主题
        def match_topic(topic_dict: Dict[str, Any], parent_path: str = "") -> None:
            for topic, subtopics in topic_dict.items():
                current_path = f"{parent_path}.{topic}" if parent_path else topic
                
                # 关键词匹配
                for kw in keywords:
                    match_score = fuzz.ratio(kw, topic)
                    if match_score >= min_threshold:
                        match_results[current_path] += match_score
                
                # 递归处理子主题
                if isinstance(subtopics, dict) and subtopics:
                    match_topic(subtopics, current_path)
        
        match_topic(self.topic_hierarchy.get("topics", {}))
        
        # 整理结果
        return [
            {
                "topic_path": path,
                "similarity": score / 100.0,
                "method": "traditional"
            }
            for path, score in sorted(match_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]

# --------------------------
# 主函数
# --------------------------
def main() -> None:
    """主函数：协调所有处理流程"""
    print("="*50)
    print("    内容主题映射工具")
    print("="*50)
    
    # 配置参数（集中管理所有可配置项）
    config = {
        # 路径配置
        "paths": {
                "json_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_contents",
                "output_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_embeded_contents",
                "hierarchy_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/config_updates/updated_config.json",
                "progress_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_embeded_contents/progress_embeded_classified.jsonl",
                "temp_keywords_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_embeded_contents/temp_keywords.jsonl",
                "update_dir": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_embeded_contents/updates",
                "stopwords_file": "/home/baobaodae/Develop/RAG_Data/Version1.0/test/classified_embeded_contents/stopwords.txt"
        },
        
        # 处理配置
        "processing": {
            "batch_size": 5,
            "max_retries": 3,
            "min_text_length": 50,
            "delete_progress_after_complete": False,
            "max_total_duration": 3600,
            "max_batch_duration": 600,
            "max_file_duration": 120,
            "batch_interval": 0.1,
            "encoding": "utf-8"
        },
        
        # 模型配置
        "model": {
            "type": "ollama",  # "ollama" / "api" / "traditional"
            "min_embedding_similarity": 0.5,
            "params": {
                # Ollama参数
                "host": "http://localhost:11434",
                "model_name": "llama3:latest",
                "stream": False,
                "timeout": 30,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                
                # API参数
                "api_url": "https://api.siliconflow.cn/v1/chat/completions",
                "api_key": "",
                "embedding_url": "https://api.siliconflow.cn/v1/embeddings",
                "embedding_model": "text-embedding-ada-002",
                "max_tokens": 100,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        },
        
        # 传统方法配置
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
        
        # 主题提取配置
        "topic_extraction": {
            "top_k_keywords": 7,
            "min_similarity": 0.3
        },
        
        # 自动更新配置
        "auto_update": {
            "enable": True,
            "min_term_frequency": 10,
            "min_similarity_threshold": 50
        },
        
        # 日志配置
        "logging": {
            "level": "INFO",
            "format": '%(asctime)s - %(levelname)s - %(message)s',
            "file": "/home/baobaodae/Develop/RAG_Data/Version1.0/topic_mapping.log"
        },
        
        # 默认停用词
        "stopwords": [
            "证", "券", "研", "究", "报", "告", "公", "司", "页", "资料来源",
            "wind", "同花顺", "东方财富", "日期", "单位", "亿元", "万元", "图表",
            "制表", "如下", "所示", "注", "注释", "摘要", "关键词", "引言", "结论"
        ],
        
        # 结果文件名称
        "result_filename": "topic_mappings_result.json"
    }
    
    # 设置日志
    global logger
    logger = setup_logging(config)
    
    try:
        # 初始化文件加载器
        file_loader = FileLoader(config)
        
        # 加载进度
        progress = file_loader.load_progress()
        completed_ids = set(progress.get("completed", []))
        failed_ids = set(progress.get("failed", []))
        
        # 加载待处理数据
        processing_data = file_loader.load_processing_data()
        
        # 过滤已处理的数据
        unprocessed_data = [
            item for item in processing_data 
            if item["id"] not in completed_ids and item["id"] not in failed_ids
        ]
        
        logger.info(f"总数据量: {len(processing_data)}, 未处理: {len(unprocessed_data)}, "
                   f"已完成: {len(completed_ids)}, 已失败: {len(failed_ids)}")
        
        if not unprocessed_data:
            logger.info("所有数据已处理完毕")
            return
        
        # 初始化处理器
        processor = TopicMappingProcessor(config)
        
        # 批处理
        batch_size = config["processing"]["batch_size"]
        total_batches = (len(unprocessed_data) + batch_size - 1) // batch_size
        final_results = {
            "topics": defaultdict(lambda: {"contents": [], "info": {}}),
            "all_contents": {},
            "uncategorized": [],
            "failed": progress.get("failed", []),
            "processing_metadata": {
                "start_time": datetime.now().isoformat(),
                "model_type": config["model"]["type"],
                "model_name": config["model"]["params"].get("model_name", "unknown")
            },
            "file_stats": defaultdict(int)
        }
        
        # 合并已有主题信息
        for topic_path, info in processor.topic_hierarchy["topic_info"].items():
            final_results["topics"][topic_path]["info"] = info
        
        # 处理所有批次
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(unprocessed_data))
            batch_data = unprocessed_data[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}，共 {len(batch_data)} 条内容")
            
            # 处理批次
            batch_results, batch_progress = processor.process_batch(batch_data)
            
            # 合并结果
            for topic_path, data in batch_results["topics"].items():
                final_results["topics"][topic_path]["contents"].extend(data["contents"])
            
            final_results["all_contents"].update(batch_results["all_contents"])
            final_results["uncategorized"].extend(batch_results["uncategorized"])
            final_results["failed"].extend(batch_results["failed"])
            
            # 更新文件统计
            for item in batch_data:
                final_results["file_stats"][item["filename"]] += 1
            
            # 保存进度
            progress["completed"].extend(batch_progress["completed"])
            progress["failed"].extend(batch_progress["failed"])
            progress["last_processed"] = datetime.now().isoformat()
            file_loader.save_progress(progress)
            
            # 批次间隔
            time.sleep(config["processing"].get("batch_interval", 0.1))
        
        # 完成处理
        final_results["processing_metadata"]["end_time"] = datetime.now().isoformat()
        final_results["file_stats"] = dict(final_results["file_stats"])
        
        # 保存最终结果
        result_file = file_loader.save_results(final_results)
        
        # 输出统计信息
        print("\n" + "="*50)
        print("处理完成，统计信息:")
        print(f"- 总处理内容: {len(processing_data)}")
        print(f"- 本次处理: {len(unprocessed_data)}")
        print(f"- 成功分类: {len(final_results['all_contents']) - len(final_results['uncategorized'])}")
        print(f"- 未分类: {len(final_results['uncategorized'])}")
        print(f"- 处理失败: {len(final_results['failed'])}")
        print(f"- 结果文件: {result_file}")
        print("="*50)
        
        # 如果配置要求，处理完成后删除进度文件
        if config["processing"].get("delete_progress_after_complete", False):
            progress_file = config["paths"].get("progress_file")
            if progress_file and Path(progress_file).exists():
                Path(progress_file).unlink()
                logger.info("处理完成，已删除进度文件")
                
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        print(f"\n处理失败: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
    