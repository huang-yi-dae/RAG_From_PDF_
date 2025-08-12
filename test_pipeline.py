import os
import json
import time
from datetime import datetime
from pipeline_manager import PipelineManager, create_default_config
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_pipeline')

def test_full_pipeline():
    """测试完整的pipeline流程"""
    logger.info("开始测试完整的pipeline流程")

    # 创建默认配置
    config = create_default_config()
    logger.info(f"使用配置: {json.dumps(config, ensure_ascii=False, indent=2)}")

    # 初始化管道管理器
    pipeline = PipelineManager(config)

    # 查找测试PDF文件
    test_dir = 'test/data_test'
    pdf_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        logger.error(f"在{test_dir}目录下未找到PDF文件")
        return {'status': 'failed', 'error': 'No PDF files found'}

    # 选择第一个PDF文件进行测试
    pdf_path = os.path.join(test_dir, pdf_files[0])
    logger.info(f"使用测试PDF文件: {pdf_path}")

    # 运行pipeline
    start_time = time.time()
    result = pipeline.run_pipeline(pdf_path)
    end_time = time.time()

    logger.info(f"pipeline运行结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    logger.info(f"pipeline运行耗时: {end_time - start_time:.2f}秒")

    return result

def test_restart_from_breakpoint():
    """测试从断点重新运行pipeline"""
    logger.info("开始测试从断点重新运行pipeline")

    # 创建默认配置
    config = create_default_config()

    # 初始化管道管理器
    pipeline = PipelineManager(config)

    # 查找测试PDF文件
    test_dir = 'test/data_test'
    pdf_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        logger.error(f"在{test_dir}目录下未找到PDF文件")
        return {'status': 'failed', 'error': 'No PDF files found'}

    # 选择第一个PDF文件进行测试
    pdf_path = os.path.join(test_dir, pdf_files[0])
    logger.info(f"使用测试PDF文件: {pdf_path}")

    # 假设我们想从步骤2开始重新运行
    restart_step = 2
    logger.info(f"从步骤{restart_step}开始重新运行pipeline")

    # 运行pipeline
    start_time = time.time()
    result = pipeline.run_pipeline(pdf_path, restart_from=restart_step)
    end_time = time.time()

    logger.info(f"pipeline重新运行结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    logger.info(f"pipeline重新运行耗时: {end_time - start_time:.2f}秒")

    return result

def test_parameter_change_detection():
    """测试参数变化检测功能"""
    logger.info("开始测试参数变化检测功能")

    # 创建默认配置
    config = create_default_config()

    # 初始化管道管理器
    pipeline = PipelineManager(config)

    # 查找测试PDF文件
    test_dir = 'test/data_test'
    pdf_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        logger.error(f"在{test_dir}目录下未找到PDF文件")
        return {'status': 'failed', 'error': 'No PDF files found'}

    # 选择第一个PDF文件进行测试
    pdf_path = os.path.join(test_dir, pdf_files[0])
    logger.info(f"使用测试PDF文件: {pdf_path}")

    # 首次运行pipeline
    logger.info("首次运行pipeline")
    result1 = pipeline.run_pipeline(pdf_path)
    logger.info(f"首次运行结果: {json.dumps(result1, ensure_ascii=False, indent=2)}")

    # 修改模型参数
    logger.info("修改模型参数")
    config['model']['params']['temperature'] = 0.7
    pipeline = PipelineManager(config)

    # 再次运行pipeline，应该重新执行受参数影响的步骤
    logger.info("修改参数后再次运行pipeline")
    result2 = pipeline.run_pipeline(pdf_path)
    logger.info(f"修改参数后运行结果: {json.dumps(result2, ensure_ascii=False, indent=2)}")

    return {'first_run': result1, 'second_run': result2}

if __name__ == '__main__':
    logger.info("===== 开始pipeline测试 =====")

    # 测试完整流程
    logger.info("===== 测试1: 完整流程 =====")
    full_result = test_full_pipeline()

    # 等待一段时间
    time.sleep(2)

    # 测试断点续跑
    logger.info("===== 测试2: 断点续跑 =====")
    restart_result = test_restart_from_breakpoint()

    # 等待一段时间
    time.sleep(2)

    # 测试参数变化检测
    logger.info("===== 测试3: 参数变化检测 =====")
    param_change_result = test_parameter_change_detection()

    logger.info("===== 测试完成 =====")

    # 汇总结果
    summary = {
        'full_pipeline': full_result,
        'restart_from_breakpoint': restart_result,
        'parameter_change': param_change_result
    }

    # 保存结果到文件
    with open('pipeline_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"测试结果已保存到 pipeline_test_results.json")