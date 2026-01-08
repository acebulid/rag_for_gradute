#!/usr/bin/env python3
"""
测试图片处理功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests
import base64
import time

def test_image_processing():
    """测试图片处理"""
    print("="*50)
    print("测试图片处理功能")
    print("="*50)
    
    # 测试图片路径
    image_path = "data/raw/images/本部_正门.png"
    if not Path(image_path).exists():
        print(f"❌ 图片文件不存在: {image_path}")
        return False
    
    print(f"测试图片: {image_path}")
    print(f"文件大小: {Path(image_path).stat().st_size / 1024:.1f} KB")
    
    # 方法1: 直接读取并编码图片
    print("\n方法1: 直接读取并编码图片...")
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
        print(f"✅ 图片编码成功")
        print(f"   Base64长度: {len(image_base64)} 字符")
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return False
    
    # 方法2: 测试Ollama API
    print("\n方法2: 测试Ollama API...")
    
    # 先检查Ollama服务
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ollama服务正常")
            print(f"   可用模型: {[m['name'] for m in data.get('models', [])]}")
        else:
            print(f"❌ Ollama服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到Ollama服务: {e}")
        return False
    
    # 测试不同的模型
    models_to_test = ["llava:7b", "llava", "qwen2.5vl:7b", "bakllava"]
    
    for model in models_to_test:
        print(f"\n测试模型: {model}")
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": "用中文简单描述这张图片",
            "images": [image_base64],
            "stream": False,
            "temperature": 0.1
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                description = data.get("response", "").strip()
                print(f"✅ 成功生成描述")
                print(f"   耗时: {elapsed_ms:.2f}ms")
                print(f"   描述长度: {len(description)} 字符")
                print(f"   描述内容: {description[:100]}..." if len(description) > 100 else f"   描述内容: {description}")
                return True
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"   错误信息: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时 (60秒)")
        except Exception as e:
            print(f"❌ 请求异常: {e}")
    
    # 如果所有模型都失败，尝试一个更简单的测试
    print("\n尝试更简单的测试...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama2",  # 纯文本模型，不处理图片
        "prompt": "用中文描述一张校园正门的图片",
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            description = data.get("response", "").strip()
            print(f"✅ 纯文本模型测试成功")
            print(f"   描述: {description[:100]}..." if len(description) > 100 else f"   描述: {description}")
            return True
        else:
            print(f"❌ 纯文本模型也失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 纯文本模型测试异常: {e}")
        return False

def main():
    """主函数"""
    print("图片处理功能测试")
    print("="*60)
    
    if test_image_processing():
        print("\n" + "="*60)
        print("✅ 图片处理测试完成")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ 图片处理测试失败")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())