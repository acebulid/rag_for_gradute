#!/usr/bin/env python3
"""
模型操作脚本 - 调用model_function演示模型功能
注意：本文件只包含主函数，不创建新函数
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import torch
import sklearn
from sklearn.preprocessing import StandardScaler

# ========== 新增：修复PyTorch 2.6+的安全加载问题 ==========
# 注册安全全局对象
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            sklearn.preprocessing._data.StandardScaler,
            StandardScaler
        ])
    # 全局设置weights_only默认值（备选方案）
    import torch.serialization
    torch.serialization.DEFAULT_LOAD_WEIGHTS_ONLY = False
except Exception as e:
    print(f"安全全局注册警告: {e}")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入model_function中的现有函数
from function.model_function import (
    ModelAConfig,
    create_model_instance,
    train_model,
    save_model,
    load_model,
    predict_single,
    batch_predict,
    evaluate_model,
    fix_model_training_status,
    verify_model_fix
)


def main():
    """
    主函数：演示模型A的完整功能
    """
    print("=" * 70)
    print("模型A功能演示")
    print("=" * 70)
    
    # 1. 创建模型配置
    config = ModelAConfig(
        input_dim=512,          
        output_dim=1024,        
        epochs=5,               
        batch_size=8,           
        print_every_n_epochs=1,
        model_path="data/models/demo_model_a.pth"
    )
    
    # 2. 创建示例数据（演示用）
    n_samples = 100
    image_embeddings = np.random.randn(n_samples, config.input_dim).astype(np.float32)
    text_embeddings = np.random.randn(n_samples, config.output_dim).astype(np.float32)
    
    # 归一化数据
    for i in range(n_samples):
        image_embeddings[i] = image_embeddings[i] / np.linalg.norm(image_embeddings[i])
        text_embeddings[i] = text_embeddings[i] / np.linalg.norm(text_embeddings[i])
    
    # 3. 训练模型
    print("\n1. 训练模型...")
    try:
        model, train_history, scaler = train_model(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            config=config,
            random_seed=42,
            verbose=False  # 关闭训练过程的详细打印
        )
        print(f"   训练完成 | 最佳验证损失: {train_history['best_val_loss']:.6f} | 耗时: {train_history['training_time']:.2f}s")
    except Exception as e:
        print(f"   训练失败: {e}")
        return
    
    # 4. 保存模型
    print("\n2. 保存模型...")
    try:
        save_model(model, scaler, config)
        print(f"   模型已保存: {config.model_path}")
    except Exception as e:
        print(f"   保存失败: {e}")
    
    # 5. 加载模型并测试预测
    print("\n3. 加载模型并预测...")
    try:
        loaded_model, loaded_scaler, device = load_model(config)
        
        # 单样本预测
        test_image = np.random.randn(config.input_dim).astype(np.float32)
        test_image = test_image / np.linalg.norm(test_image)
        meaning_vector = predict_single(loaded_model, loaded_scaler, device, test_image, verbose=False)
        
        # 批量预测
        batch_images = np.random.randn(5, config.input_dim).astype(np.float32)
        for i in range(5):
            batch_images[i] = batch_images[i] / np.linalg.norm(batch_images[i])
        batch_vectors = batch_predict(loaded_model, loaded_scaler, device, batch_images, verbose=False)
        
        print(f"   加载成功 | 设备: {device} | 单样本输出维度: {len(meaning_vector)} | 批量输出形状: {batch_vectors.shape}")
    except Exception as e:
        print(f"   加载/预测失败: {e}")
    
    # 6. 模型状态修复和验证
    print("\n4. 模型状态修复与验证...")
    try:
        # 修复模型状态
        fix_result = fix_model_training_status(config.model_path, True, config)
        if fix_result["success"]:
            # ========== 关键修改1：修正参数传递顺序 ==========
            # 准备测试向量（和步骤5中用的测试向量一致）
            test_vector = np.random.randn(config.input_dim).astype(np.float32)
            test_vector = test_vector / np.linalg.norm(test_vector)
            
            # ========== 关键修改2：打开verbose=True，查看详细验证日志 ==========
            verify_result = verify_model_fix(
                model_path=config.model_path,
                test_vector=test_vector,  # 正确传递测试向量
                config=config,            # 正确传递config参数
                verbose=True              # 开启详细日志
            )
            
            # 打印完整的验证结果（便于调试）
            print(f"\n   完整验证结果:")
            print(f"      - 模型路径: {verify_result['model_path']}")
            print(f"      - 训练状态: {verify_result['current_training_status']}")
            print(f"      - 预测成功: {verify_result['prediction_test'].get('prediction_success', False)}")
            print(f"      - 验证成功: {verify_result['verification_success']}")
            if verify_result['error']:
                print(f"      - 错误信息: {verify_result['error']}")
            
            print(f"\n   修复成功 | 验证{'通过' if verify_result['verification_success'] else '失败'}")
        else:
            print(f"   修复失败: {fix_result.get('error', '未知错误')}")
    except Exception as e:
        print(f"   状态修复/验证失败: {e}")
        print(f"   详细错误栈: {traceback.format_exc()}")
    
    print("\n" + "=" * 70)
    print("模型演示完成")
    print("=" * 70)


if __name__ == "__main__":
    # 修复dataclass导入问题
    if sys.version_info >= (3, 7):
        from dataclasses import dataclass
    else:
        def dataclass(cls):
            return cls
    
    # 重新应用dataclass装饰器
    from function.model_function import ModelAConfig as OriginalModelAConfig
    globals()['ModelAConfig'] = dataclass(OriginalModelAConfig)
    
    main()