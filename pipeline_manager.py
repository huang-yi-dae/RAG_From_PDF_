import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# 导入项目中的其他模块
from pdf_element_extractor import extract_elements_from_pdf
from content_classifier import load_topic_hierarchy, extract_keywords, call_api_model
from embedding_generator import BaseModelProcessor, OllamaModelProcessor, ApiModelProcessor
from rag_engine import get_embedding, preprocess_text
from system_monitor import get_cpu_usage, get_memory_usage, get_gpu_usage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline_manager')

class PipelineManager:
    """管道管理器，负责协调各个处理步骤并提供监控和断点续跑功能"""
    def __init__(self, config: Dict[str, Any]):
        """初始化管道管理器

        Args:
            config: 包含所有管道配置的字典
        """
        self.config = config
        self.status_file = config.get('status_file', 'pipeline_status.json')
        self.task_hash_file = config.get('task_hash_file', 'task_hash.json')
        self.model_processor = self._initialize_model_processor()
        self.topic_hierarchy = self._load_topic_hierarchy()
        self.status = self._load_status()
        self.task_hashes = self._load_task_hashes()

    def _initialize_model_processor(self) -> BaseModelProcessor:
        """初始化模型处理器"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'ollama')

        if model_type == 'ollama':
            return OllamaModelProcessor(model_config)
        elif model_type == 'api':
            return ApiModelProcessor(model_config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _load_topic_hierarchy(self) -> Dict[str, List[str]]:
        """加载主题分类体系"""
        topic_file = self.config.get('topic_file', 'topic_hierarchy.json')
        try:
            return load_topic_hierarchy(topic_file)
        except Exception as e:
            logger.error(f"加载主题分类体系失败: {str(e)}")
            return {}

    def _load_status(self) -> Dict[str, Any]:
        """加载上次运行的状态"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载状态文件失败: {str(e)}")
        return {
            'last_run': None,
            'current_step': 0,
            'steps': [],
            'errors': [],
            'completed_tasks': {}
        }

    def _load_task_hashes(self) -> Dict[str, str]:
        """加载任务参数哈希，用于检测参数变化"""
        if os.path.exists(self.task_hash_file):
            try:
                with open(self.task_hash_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载任务哈希文件失败: {str(e)}")
        return {}

    def _save_status(self) -> None:
        """保存当前状态"""
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存状态文件失败: {str(e)}")

    def _save_task_hashes(self) -> None:
        """保存任务参数哈希"""
        try:
            with open(self.task_hash_file, 'w', encoding='utf-8') as f:
                json.dump(self.task_hashes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存任务哈希文件失败: {str(e)}")

    def _generate_task_hash(self, task_name: str, params: Dict[str, Any]) -> str:
        """生成任务参数的哈希值，用于检测参数变化"""
        # 将参数字典转换为排序后的字符串
        param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        # 组合任务名和参数字符串
        combined = f"{task_name}:{param_str}"
        # 生成MD5哈希
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def _check_task_params_changed(self, task_name: str, params: Dict[str, Any]) -> bool:
        """检查任务参数是否发生变化"""
        current_hash = self._generate_task_hash(task_name, params)
        previous_hash = self.task_hashes.get(task_name)

        # 更新哈希值
        self.task_hashes[task_name] = current_hash
        self._save_task_hashes()

        # 如果之前没有哈希或者哈希不同，则参数已更改
        return previous_hash is None or previous_hash != current_hash

    def _monitor_system(self) -> Dict[str, Any]:
        """监控系统资源使用情况"""
        try:
            cpu = get_cpu_usage()
            memory = get_memory_usage()
            gpu = get_gpu_usage()

            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': cpu,
                'memory': memory,
                'gpu': gpu
            }
        except Exception as e:
            logger.error(f"监控系统资源失败: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _add_step(self, name: str, params: Dict[str, Any] = None) -> int:
        """添加一个处理步骤"""
        step_id = len(self.status['steps'])
        self.status['steps'].append({
            'id': step_id,
            'name': name,
            'params': params or {},
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'error': None,
            'system_metrics': []
        })
        self._save_status()
        return step_id

    def _start_step(self, step_id: int) -> None:
        """开始执行一个步骤"""
        if 0 <= step_id < len(self.status['steps']):
            self.status['steps'][step_id]['status'] = 'running'
            self.status['steps'][step_id]['start_time'] = datetime.now().isoformat()
            self.status['current_step'] = step_id
            self._save_status()

    def _complete_step(self, step_id: int, result: Any = None) -> None:
        """完成一个步骤"""
        if 0 <= step_id < len(self.status['steps']):
            self.status['steps'][step_id]['status'] = 'completed'
            self.status['steps'][step_id]['end_time'] = datetime.now().isoformat()
            self.status['steps'][step_id]['result'] = result
            self._save_status()

    def _fail_step(self, step_id: int, error: Exception) -> None:
        """标记一个步骤为失败"""
        if 0 <= step_id < len(self.status['steps']):
            self.status['steps'][step_id]['status'] = 'failed'
            self.status['steps'][step_id]['end_time'] = datetime.now().isoformat()
            self.status['steps'][step_id]['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
            self.status['errors'].append({
                'step_id': step_id,
                'step_name': self.status['steps'][step_id]['name'],
                'error_type': type(error).__name__,
                'error_message': str(error),
                'timestamp': datetime.now().isoformat()
            })
            self._save_status()

    def _should_skip_step(self, step_id: int, task_name: str, params: Dict[str, Any]) -> bool:
        """检查是否应该跳过某个步骤"""
        # 如果步骤已经完成，且参数没有变化，则跳过
        if (0 <= step_id < len(self.status['steps']) and 
            self.status['steps'][step_id]['status'] == 'completed' and 
            not self._check_task_params_changed(task_name, params)):
            return True
        return False

    def run_pipeline(self, pdf_path: str, restart_from: int = None) -> Dict[str, Any]:
        """运行完整的处理管道

        Args:
            pdf_path: PDF文件路径
            restart_from: 从哪个步骤开始重新运行，默认为None（从头开始）

        Returns:
            包含处理结果的字典
        """
        start_time = datetime.now()
        self.status['last_run'] = start_time.isoformat()
        self._save_status()

        try:
            # 步骤1: 从PDF中提取元素
            step1_id = self._add_step('extract_pdf_elements', {'pdf_path': pdf_path})
            if restart_from is None or step1_id >= restart_from:
                if not self._should_skip_step(step1_id, 'extract_pdf_elements', {'pdf_path': pdf_path}):
                    self._start_step(step1_id)
                    # 监控系统资源
                    metrics = self._monitor_system()
                    self.status['steps'][step1_id]['system_metrics'].append(metrics)

                    # 执行PDF元素提取
                    pdf_file_name = os.path.basename(pdf_path)
                    image_dir = self.config.get('image_dir', 'extracted_images')
                    os.makedirs(image_dir, exist_ok=True)
                    elements = extract_elements_from_pdf(pdf_path, pdf_file_name, image_dir)

                    # 完成步骤
                    self._complete_step(step1_id, {'elements_count': len(elements)})
                    logger.info(f"步骤1: PDF元素提取完成，共提取{len(elements)}个元素")
                else:
                    logger.info(f"步骤1: PDF元素提取已完成且参数未变化，跳过")
            else:
                logger.info(f"步骤1: 跳过，将从步骤{restart_from}开始")

            # 步骤2: 内容分类
            step2_id = self._add_step('classify_content', {'pdf_path': pdf_path})
            if restart_from is None or step2_id >= restart_from:
                if not self._should_skip_step(step2_id, 'classify_content', {'pdf_path': pdf_path}):
                    self._start_step(step2_id)
                    # 监控系统资源
                    metrics = self._monitor_system()
                    self.status['steps'][step2_id]['system_metrics'].append(metrics)

                    # 执行内容分类
                    # 这里需要获取步骤1的结果，如果是跳过的，则需要从之前的结果中获取
                    if step1_id < len(self.status['steps']) and self.status['steps'][step1_id]['status'] == 'completed':
                        # 实际应用中，这里应该加载之前保存的元素数据
                        # 为简化示例，我们假设元素已经在内存中
                        text_elements = [e for e in elements if e['type'] == 'text']
                        logger.info(f"找到{len(text_elements)}个文本元素进行分类")

                        # 完成步骤
                        self._complete_step(step2_id, {'classified_count': len(text_elements)})
                        logger.info(f"步骤2: 内容分类完成，共分类{len(text_elements)}个文本元素")
                    else:
                        error = Exception("无法获取PDF元素数据，步骤1未完成")
                        self._fail_step(step2_id, error)
                        logger.error(f"步骤2: 内容分类失败: {str(error)}")
                        raise error
                else:
                    logger.info(f"步骤2: 内容分类已完成且参数未变化，跳过")
            else:
                logger.info(f"步骤2: 跳过，将从步骤{restart_from}开始")

            # 步骤3: 生成嵌入向量
            step3_id = self._add_step('generate_embeddings', {'pdf_path': pdf_path})
            if restart_from is None or step3_id >= restart_from:
                if not self._should_skip_step(step3_id, 'generate_embeddings', {'pdf_path': pdf_path}):
                    self._start_step(step3_id)
                    # 监控系统资源
                    metrics = self._monitor_system()
                    self.status['steps'][step3_id]['system_metrics'].append(metrics)

                    # 执行嵌入生成
                    # 这里需要获取步骤2的结果
                    if step2_id < len(self.status['steps']) and self.status['steps'][step2_id]['status'] == 'completed':
                        # 实际应用中，这里应该加载之前保存的分类数据
                        # 为简化示例，我们假设文本元素已经在内存中
                        embeddings = []
                        for element in text_elements:
                            embedding = get_embedding(element['content'], self.config['model'])
                            if len(embedding) > 0:
                                embeddings.append({
                                    'element_id': element['id'],
                                    'embedding': embedding.tolist()
                                })

                        # 完成步骤
                        self._complete_step(step3_id, {'embeddings_count': len(embeddings)})
                        logger.info(f"步骤3: 嵌入向量生成完成，共生成{len(embeddings)}个嵌入")
                    else:
                        error = Exception("无法获取分类数据，步骤2未完成")
                        self._fail_step(step3_id, error)
                        logger.error(f"步骤3: 嵌入向量生成失败: {str(error)}")
                        raise error
                else:
                    logger.info(f"步骤3: 嵌入向量生成已完成且参数未变化，跳过")
            else:
                logger.info(f"步骤3: 跳过，将从步骤{restart_from}开始")

            # 步骤4: RAG处理
            step4_id = self._add_step('rag_processing', {'pdf_path': pdf_path})
            if restart_from is None or step4_id >= restart_from:
                if not self._should_skip_step(step4_id, 'rag_processing', {'pdf_path': pdf_path}):
                    self._start_step(step4_id)
                    # 监控系统资源
                    metrics = self._monitor_system()
                    self.status['steps'][step4_id]['system_metrics'].append(metrics)

                    # 执行RAG处理
                    # 这里需要获取步骤3的结果
                    if step3_id < len(self.status['steps']) and self.status['steps'][step3_id]['status'] == 'completed':
                        # 实际应用中，这里会执行RAG相关操作
                        logger.info("执行RAG处理...")

                        # 完成步骤
                        self._complete_step(step4_id, {'status': 'completed'})
                        logger.info("步骤4: RAG处理完成")
                    else:
                        error = Exception("无法获取嵌入数据，步骤3未完成")
                        self._fail_step(step4_id, error)
                        logger.error(f"步骤4: RAG处理失败: {str(error)}")
                        raise error
                else:
                    logger.info(f"步骤4: RAG处理已完成且参数未变化，跳过")
            else:
                logger.info(f"步骤4: 跳过，将从步骤{restart_from}开始")

            # 所有步骤完成
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"管道执行完成，总耗时: {total_time:.2f}秒")
            return {
                'status': 'success',
                'total_time': total_time,
                'steps_completed': len([s for s in self.status['steps'] if s['status'] == 'completed']),
                'errors': len(self.status['errors'])
            }

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"管道执行失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'total_time': total_time,
                'current_step': self.status['current_step'],
                'steps_completed': len([s for s in self.status['steps'] if s['status'] == 'completed'])
            }

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'status_file': 'pipeline_status.json',
        'task_hash_file': 'task_hash.json',
        'image_dir': 'extracted_images',
        'topic_file': 'topic_hierarchy.json',
        'model': {
            'type': 'ollama',
            'params': {
                'host': 'http://localhost:11434',
                'model_name': 'llama3:latest',
                'embedding_model': 'nomic-embed-text',
                'timeout': 30,
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 40
            }
        }
    }

# 示例用法
if __name__ == '__main__':
    # 创建默认配置
    config = create_default_config()

    # 初始化管道管理器
    pipeline = PipelineManager(config)

    # 运行管道（假设存在一个测试PDF文件）
    pdf_path = 'test/data_test/sample.pdf'
    if os.path.exists(pdf_path):
        result = pipeline.run_pipeline(pdf_path)
        print(f"管道运行结果: {result}")
    else:
        print(f"测试PDF文件不存在: {pdf_path}")
        print("请修改pdf_path变量指向一个存在的PDF文件")