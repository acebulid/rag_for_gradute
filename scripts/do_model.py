#!/usr/bin/env python3
"""
模型操作脚本 - 调用model_function训练图片到关键词分类模型
注意：本文件只包含主函数，不创建新函数
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import json
import torch
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ========== 修复：仅保留有效的安全全局对象注册 ==========
# 注册安全全局对象（移除无效的DEFAULT_LOAD_WEIGHTS_ONLY设置）
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            sklearn.preprocessing._data.StandardScaler,
            StandardScaler,
            sklearn.preprocessing._label.LabelEncoder,
            LabelEncoder
        ])
except Exception as e:
    print(f"安全全局注册警告: {e}")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入model_function中的现有函数
from function.model_function import (
    ModelAConfig,
    train_model,
    save_model,
    load_model,
    predict_single,
    batch_predict
)


def load_training_data():
    """
    从database_data.json加载训练数据
    返回: (image_embeddings, keyword_labels)
    """
    print("加载训练数据...")
    
    # 加载database_data.json
    data_path = Path("data/processed/database_data.json")
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        return None, None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取章节信息
    chapters = data.get('chapters', [])
    if not chapters:
        print("错误: 没有找到章节数据")
        return None, None
    
    # 收集图片向量和关键词标签
    image_embeddings = []
    keyword_labels = []
    
    print(f"找到 {len(chapters)} 个章节")
    
    # 为每个章节的图片生成向量
    for chapter in chapters:
        chapter_title = chapter.get('title', '')
        image_embeddings_data = chapter.get('image_embeddings', [])
        
        print(f"  处理章节: {chapter_title}, 图片向量数量: {len(image_embeddings_data)}")
        
        # 使用真实的图片向量
        for image_data in image_embeddings_data:
            image_embedding = image_data.get('embedding', [])
            if not image_embedding:
                print(f"    警告: 图片向量为空，跳过")
                continue
            
            # 确保向量是numpy数组
            image_vector = np.array(image_embedding, dtype=np.float32)
            
            # 归一化向量
            norm = np.linalg.norm(image_vector)
            if norm > 0:
                image_vector = image_vector / norm
            
            image_embeddings.append(image_vector)
            keyword_labels.append(chapter_title)
    
    if not image_embeddings:
        print("错误: 没有生成任何训练数据")
        return None, None
    
    image_embeddings = np.array(image_embeddings)
    
    print(f"训练数据加载完成")
    print(f"  总样本数: {len(image_embeddings)}")
    print(f"  输入维度: {image_embeddings.shape[1]}")
    print(f"  关键词类别: {set(keyword_labels)}")
    
    return image_embeddings, keyword_labels


def main():
    """
    主函数：训练图片到关键词分类模型
    """
    print("=" * 70)
    print("图片到关键词分类模型训练")
    print("=" * 70)
    
    # 1. 创建模型配置
    # 确保model文件夹存在
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    config = ModelAConfig(
        input_dim=512,           # CLIP图片向量维度（512维）
        hidden_dims=(256, 128, 64),  # 隐藏层维度相应调整
        num_classes=9,           # 9个关键词类别
        epochs=50,               # 训练轮数
        batch_size=16,           # 批次大小
        print_every_n_epochs=5,  # 每5个epoch打印一次
        model_path="model/model_a.pth",
        config_path="model/model_a_config.json",
        scaler_path="model/model_a_scaler.pkl",
        label_encoder_path="model/model_a_label_encoder.pkl"
    )
    
    # 2. 加载训练数据
    print("\n1. 加载训练数据...")
    image_embeddings, keyword_labels = load_training_data()
    
    if image_embeddings is None or keyword_labels is None:
        print("训练数据加载失败，退出")
        return
    
    # 3. 训练模型
    print("\n2. 训练模型...")
    try:
        model, train_history, scaler, label_encoder = train_model(
            image_embeddings=image_embeddings,
            keyword_labels=keyword_labels,
            config=config,
            random_seed=42,
            verbose=True  # 显示训练过程
        )
        
        print(f"   训练完成")
        print(f"   最佳验证准确率: {train_history['best_val_accuracy']:.4f}")
        print(f"   最终训练准确率: {train_history['train_accuracies'][-1]:.4f}")
        print(f"   最终验证准确率: {train_history['val_accuracies'][-1]:.4f}")
        print(f"   训练耗时: {train_history['training_time']:.2f}s")
        
    except Exception as e:
        print(f"   训练失败: {e}")
        print(f"   详细错误栈: {traceback.format_exc()}")
        return
    
    # 4. 保存模型
    print("\n3. 保存模型...")
    try:
        save_model(model, scaler, label_encoder, config)
        print(f"   模型已保存: {config.model_path}")
        print(f"   配置文件已保存: {config.config_path}")
        print(f"   Scaler已保存: {config.scaler_path}")
        print(f"   标签编码器已保存: {config.label_encoder_path}")
        
    except Exception as e:
        print(f"   保存失败: {e}")
        return
    
    # 5. 加载模型并测试预测
    print("\n4. 加载模型并测试预测...")
    try:
        loaded_model, loaded_scaler, loaded_label_encoder, device = load_model(config)
        
        print(f"   模型加载成功")
        print(f"   设备: {device}")
        print(f"   关键词类别: {loaded_label_encoder.classes_}")
        
        # 单样本预测测试
        print(f"\n   单样本预测测试:")
        test_image = image_embeddings[0]  # 使用第一个训练样本
        keyword, confidence = predict_single(
            loaded_model, loaded_scaler, loaded_label_encoder, device, 
            test_image, verbose=True
        )
        
        # 批量预测测试
        print(f"\n   批量预测测试:")
        batch_images = image_embeddings[:5]  # 使用前5个训练样本
        batch_results = batch_predict(
            loaded_model, loaded_scaler, loaded_label_encoder, device,
            batch_images, verbose=True
        )
        
        print(f"\n   批量预测结果:")
        for i, (keyword, confidence) in enumerate(batch_results, 1):
            print(f"      [{i}] 关键词: {keyword}, 置信度: {confidence:.4f}")
        
    except Exception as e:
        print(f"   加载/预测失败: {e}")
        print(f"   详细错误栈: {traceback.format_exc()}")
    
    print("\n" + "=" * 70)
    print("模型训练完成")
    print("=" * 70)
    
    # 6. 提供使用说明
    print("\n使用说明:")
    print("1. 重新训练模型: python scripts/do_model.py")
    print("2. 测试模型预测: python scripts/do_model.py --test")
    print("3. 查看模型信息: python scripts/do_model.py --info")


def test_model():
    """
    测试已训练的模型
    """
    print("=" * 70)
    print("测试已训练的模型")
    print("=" * 70)
    
    # 加载模型配置
    config = ModelAConfig()
    
    # 加载模型
    try:
        model, scaler, label_encoder, device = load_model(config)
        
        print(f"模型加载成功")
        print(f"设备: {device}")
        print(f"关键词类别: {label_encoder.classes_}")
        print(f"类别数量: {len(label_encoder.classes_)}")
        
        # 生成随机测试向量
        print(f"\n随机测试:")
        for i in range(3):
            test_vector = np.random.randn(config.input_dim).astype(np.float32)
            test_vector = test_vector / np.linalg.norm(test_vector)
            
            keyword, confidence = predict_single(
                model, scaler, label_encoder, device, test_vector, verbose=True
            )
        
        # 测试真实图片向量（如果有）
        print(f"\n真实数据测试:")
        image_embeddings, keyword_labels = load_training_data()
        if image_embeddings is not None and len(image_embeddings) > 0:
            # 随机选择5个样本
            indices = np.random.choice(len(image_embeddings), min(5, len(image_embeddings)), replace=False)
            for idx in indices:
                test_vector = image_embeddings[idx]
                true_keyword = keyword_labels[idx]
                
                keyword, confidence = predict_single(
                    model, scaler, label_encoder, device, test_vector, verbose=True
                )
                print(f"   真实关键词: {true_keyword}, 预测关键词: {keyword}, 置信度: {confidence:.4f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print(f"详细错误栈: {traceback.format_exc()}")


def show_model_info():
    """
    显示模型信息
    """
    print("=" * 70)
    print("模型信息")
    print("=" * 70)
    
    # 加载模型配置
    config = ModelAConfig()
    
    # ========== 新增：多文件存在性校验 ==========
    required_files = [
        (config.model_path, "模型文件"),
        (config.config_path, "配置文件"),
        (config.scaler_path, "Scaler文件"),
        (config.label_encoder_path, "标签编码器文件")
    ]
    
    for file_path, file_desc in required_files:
        path = Path(file_path)
        if not path.exists():
            print(f"错误: {file_desc}不存在: {path}")
            return
    
    try:
        # 加载模型
        model, scaler, label_encoder, device = load_model(config)
        
        print(f"模型文件: {config.model_path}")
        print(f"配置文件: {config.config_path}")
        print(f"Scaler文件: {config.scaler_path}")
        print(f"标签编码器文件: {config.label_encoder_path}")
        print(f"计算设备: {device}")
        print(f"输入维度: {config.input_dim}")
        print(f"隐藏层维度: {config.hidden_dims}")
        print(f"输出类别数: {config.num_classes}")
        print(f"关键词类别: {label_encoder.classes_}")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
    except Exception as e:
        print(f"加载模型信息失败: {e}")


def parse_arguments():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='模型操作脚本')
    parser.add_argument('--test', action='store_true', help='测试已训练的模型')
    parser.add_argument('--info', action='store_true', help='显示模型信息')
    
    return parser.parse_args()


if __name__ == "__main__":
    # ========== 修复：删除重复的dataclass装饰器 ==========
    # 直接解析命令行参数，无需重新装饰ModelAConfig
    args = parse_arguments()
    
    if args.test:
        test_model()
    elif args.info:
        show_model_info()
    else:
        main()
