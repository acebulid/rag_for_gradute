#!/usr/bin/env python3
"""
RAG系统主程序
集成关键词向量检索与本地模型生成能力
"""

import asyncio
import sys
from pathlib import Path

# 项目根目录加入Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def main():
    """
    RAG系统核心工作流
    
    执行步骤：
    1.  初始化 Ollama 服务与向量数据库连接
    2.  获取用户提问
    3.  从提问中提取核心关键词
    4.  将关键词转化为向量表示
    5.  在向量数据库中检索相似文档
    6.  选取相似度最高的文档作为上下文
    7.  拼接上下文与原始问题，发送给本地大模型
    8.  接收并展示模型生成的回答
    9.  记录本次查询的完整历史
    10. 关闭所有服务连接，释放资源
    
    设计亮点：
    - 异常捕获：覆盖 Ollama 超时、数据库连接中断等场景
    - 性能优化：减少大模型的无效调用
    - 灵活扩展：预留批量处理、交互模式等入口
    - 反馈明确：每一步都有进度提示和状态反馈
    """
    
    print("=" * 70)
    print("RAG系统 - 关键词向量检索 + 本地模型生成")
    print("=" * 70)
    
    try:
        # --------------------------
        # 1. 服务初始化
        # --------------------------
        print("\n[1/6] 初始化服务...")
        
        # 导入依赖模块
        # from src.services.ollama_service import OllamaService
        # from src.database.vector_store import PostgreSQLVectorStore
        
        # 启动 Ollama 服务
        # ollama_service = OllamaService(model=CONFIG["ollama_model"])
        
        # 连接向量数据库
        # vector_store = PostgreSQLVectorStore()
        # await vector_store.connect()
        
        print("   服务初始化完成")
        
        # --------------------------
        # 2. 获取用户提问
        # --------------------------
        print("\n[2/6] 获取用户提问...")
        
        # 可从命令行参数或交互输入获取，此处使用示例提问
        question = "首都师范大学本部正门在哪里？"
        
        print(f"   提问: {question}")
        
        # --------------------------
        # 3. 提取关键词
        # --------------------------
        print("\n[3/6] 提取关键词...")
        
        # 策略：简单分词后提取核心实体
        # 备选：调用 Ollama 做智能提取（需处理超时）
        keywords = ['首都师范大学', '本部', '正门']
        
        print(f"   提取到的关键词: {keywords}")
        
        # --------------------------
        # 4. 生成关键词向量
        # --------------------------
        print("\n[4/6] 生成关键词向量...")
        
        # 优化：复用主关键词向量，减少计算量
        keyword_embeddings = []
        
        print(f"   生成 {len(keyword_embeddings)} 个关键词向量")
        
        # --------------------------
        # 5. 向量检索
        # --------------------------
        print("\n[5/6] 关键词向量检索...")
        
        # 在向量库中检索相似文档
        # search_results = await vector_store.search_by_keywords(
        #     keyword_embeddings,
        #     top_k=CONFIG["top_k"],
        #     threshold=CONFIG["similarity_threshold"]
        # )
        
        search_results = []  # 示例结果
        
        print(f"   找到 {len(search_results)} 个相关文档")
        
        # --------------------------
        # 6. 处理检索结果
        # --------------------------
        if search_results:
            # 取相似度最高的文档
            top_doc = search_results[0]
            doc_content = top_doc.content
            doc_similarity = top_doc.score
            
            print(f"\n   最相关文档 (相似度: {doc_similarity:.4f}):")
            print(f"   内容预览: {doc_content[:100]}...")
            
            # --------------------------
            # 7. 调用本地模型生成回答
            # --------------------------
            print("\n[6/6] 生成回答...")
            
            # 构建提示词
            # prompt = f"基于以下文档内容回答问题: {doc_content}\n\n问题: {question}"
            # response = await ollama_service.generate_text(prompt)
            
            model_response = "这是模型生成的示例回答..."
            
            print(f"\n   {'='*50}")
            print(f"   模型回答:")
            print(f"   {'='*50}")
            print(f"   {model_response}")
            print(f"   {'='*50}")
            
            # --------------------------
            # 8. 记录查询历史
            # --------------------------
            # await vector_store.log_query_history(
            #     query_text=question,
            #     retrieved_document_ids=[top_doc.id],
            #     response=model_response
            # )
            
            print(f"\n   查询历史已记录")
            
        else:
            print(f"\n   未找到相关文档")
            
        # --------------------------
        # 9. 资源清理
        # --------------------------
        print("\n[清理] 关闭资源...")
        
        # await vector_store.close()
        
        print("   资源已清理")
        
        # --------------------------
        # 10. 完成
        # --------------------------
        print("\n" + "=" * 70)
        print("RAG系统处理完成!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n系统处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# --------------------------
# 扩展功能，暂时不实现
# --------------------------
async def batch_process_questions(questions):
    """批量处理多个提问"""
    pass


async def interactive_mode():
    """交互式问答模式"""
    pass


async def test_mode():
    """运行测试用例"""
    pass


# --------------------------
# 全局配置
# --------------------------
CONFIG = {
    "ollama_model": "qwen3:8b",
    "embedding_model": "bge-m3",
    "top_k": 5,
    "similarity_threshold": 0.3,
    "max_keywords": 5,
    "enable_history": True,
    "enable_logging": True,
}


# --------------------------
# 程序入口
# --------------------------
if __name__ == "__main__":
    """
    RAG系统主程序 - 使用方法
    
    本程序支持以下功能模式：
    
    1.  基础RAG工作流：
        python main_function_draft.py --mode rag
        
        功能：执行完整的RAG工作流，包括关键词提取、向量检索、本地模型生成回答
        
    2.  训练模型A：
        python main_function_draft.py --mode train-model-a
        
        功能：训练模型A（图片编码 → 含义向量）
        训练过程将显示详细的训练效果：
        - 每个epoch的训练损失和验证损失
        - 训练/验证损失比
        - 每个epoch的耗时
        - 最佳模型保存信息
        - 训练完成后的评估结果
        
        训练步骤：
        1. 导入模型A模块
        2. 加载训练数据（图片向量和文字向量）
        3. 配置模型参数（输入维度、隐藏层、输出维度等）
        4. 创建模型实例
        5. 开始训练，每N个epoch打印训练效果
        6. 保存最佳模型
        7. 评估模型性能
        8. 测试模型调用
        
    3.  调用模型A：
        python main_function_draft.py --mode use-model-a
        
        功能：在RAG工作流中调用已训练的模型A
        调用过程将显示模型工作情况：
        - 模型加载状态
        - 输入向量维度
        - 输出向量维度
        - 推理耗时
        - 累计调用次数
        
        调用步骤：
        1. 初始化Ollama服务和向量数据库
        2. 加载已训练的模型A
        3. 获取用户提问
        4. 提取关键词
        5. 生成关键词向量
        6. 使用关键词向量进行检索
        7. 获取最相关文档
        8. 调用本地模型生成回答
        9. 显示回答内容
        10. 记录查询历史
        
    4.  测试完整工作流：
        python main_function_draft.py --mode test-all
        
        功能：测试模型A的完整工作流
        包括：训练 → 调用 → RAG集成
        
    5.  交互模式：
        python main_function_draft.py --mode interactive
        
        功能：进入交互式问答模式
        用户可以连续提问，系统实时回答
        
    6.  批量处理：
        python main_function_draft.py --mode batch --input questions.txt
        
        功能：批量处理文件中的多个问题
        
    参数说明：
        --mode: 运行模式（rag, train-model-a, use-model-a, test-all, interactive, batch）
        --question: 单个问题（用于rag或use-model-a模式）
        --input: 输入文件路径（用于batch模式）
        --epochs: 训练epoch数（用于train-model-a模式）
        --batch-size: 训练批次大小（用于train-model-a模式）
        
    示例：
        训练模型A（50个epoch）：
        python main_function_draft.py --mode train-model-a --epochs 50
        
        使用模型A回答特定问题：
        python main_function_draft.py --mode use-model-a --question "首都师范大学本部正门在哪里？"
        
        执行完整RAG工作流：
        python main_function_draft.py --mode rag
        
        测试完整工作流：
        python main_function_draft.py --mode test-all
    """
    
    # 解析命令行参数
    # import argparse
    # parser = argparse.ArgumentParser(description='RAG系统主程序')
    # parser.add_argument('--mode', type=str, default='rag', 
    #                    choices=['rag', 'train-model-a', 'use-model-a', 'test-all', 'interactive', 'batch'],
    #                    help='运行模式')
    # parser.add_argument('--question', type=str, help='用户提问')
    # parser.add_argument('--input', type=str, help='输入文件路径（用于batch模式）')
    # parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    # parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
    # parser.add_argument('--interactive', action='store_true', help='交互模式')
    # parser.add_argument('--test', action='store_true', help='测试模式')
    # args = parser.parse_args()
    
    # 根据模式执行不同的功能
    # if args.mode == 'rag':
    #     # 执行基础RAG工作流
    #     success = asyncio.run(main())
    # elif args.mode == 'train-model-a':
    #     # 训练模型A
    #     success = asyncio.run(train_model_a(args.epochs, args.batch_size))
    # elif args.mode == 'use-model-a':
    #     # 调用模型A
    #     question = args.question or "首都师范大学本部正门在哪里？"
    #     success = asyncio.run(use_model_a_in_rag(question))
    # elif args.mode == 'test-all':
    #     # 测试完整工作流
    #     success = asyncio.run(test_model_a_workflow())
    # elif args.mode == 'interactive':
    #     # 交互模式
    #     success = asyncio.run(interactive_mode())
    # elif args.mode == 'batch':
    #     # 批量处理
    #     success = asyncio.run(batch_process_questions(args.input))
    # else:
    #     print(f"未知模式: {args.mode}")
    #     success = False
    
    # 当前仅执行基础RAG工作流（演示用）
    success = asyncio.run(main())
    
    if success:
        print("\n程序执行成功!")
        sys.exit(0)
    else:
        print("\n程序执行失败!")
        sys.exit(1)
