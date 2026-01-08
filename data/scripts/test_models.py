#!/usr/bin/env python3
"""
数据库模型测试
测试SQLAlchemy模型定义和导入
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_models_import():
    """测试模型导入"""
    print("="*50)
    print("测试数据库模型导入")
    print("="*50)
    
    try:
        from src.database.models import (
            Base, Document, ImageDescription, 
            TextImageRelation, QueryHistory
        )
        
        print("✅ 所有模型导入成功")
        print(f"   基础类: {Base}")
        print(f"   文档模型: {Document}")
        print(f"   图像描述模型: {ImageDescription}")
        print(f"   文本-图像关联模型: {TextImageRelation}")
        print(f"   查询历史模型: {QueryHistory}")
        
        return True
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def test_model_structure():
    """测试模型结构"""
    print("\n" + "="*50)
    print("测试模型结构")
    print("="*50)
    
    try:
        from src.database.models import Document, ImageDescription
        
        # 检查Document模型
        print("📄 Document模型结构:")
        print(f"   表名: {Document.__tablename__}")
        print(f"   列: {[col.name for col in Document.__table__.columns]}")
        
        # 检查ImageDescription模型
        print("\n🖼️  ImageDescription模型结构:")
        print(f"   表名: {ImageDescription.__tablename__}")
        print(f"   列: {[col.name for col in ImageDescription.__table__.columns]}")
        
        return True
    except Exception as e:
        print(f"❌ 模型结构检查失败: {e}")
        return False

def test_sqlalchemy_metadata():
    """测试SQLAlchemy元数据"""
    print("\n" + "="*50)
    print("测试SQLAlchemy元数据")
    print("="*50)
    
    try:
        from src.database.models import Base
        
        print("✅ SQLAlchemy元数据检查:")
        print(f"   表数量: {len(Base.metadata.tables)}")
        
        for table_name, table in Base.metadata.tables.items():
            print(f"   表: {table_name}")
            print(f"     列数: {len(table.columns)}")
            print(f"     主键: {[pk.name for pk in table.primary_key]}")
        
        return True
    except Exception as e:
        print(f"❌ 元数据检查失败: {e}")
        return False

def test_model_relationships():
    """测试模型关系"""
    print("\n" + "="*50)
    print("测试模型关系")
    print("="*50)
    
    try:
        from src.database.models import TextImageRelation
        
        # 检查外键关系
        print("🔗 TextImageRelation模型关系:")
        for fk in TextImageRelation.__table__.foreign_keys:
            print(f"   外键: {fk.column.name} -> {fk.target_fullname}")
        
        return True
    except Exception as e:
        print(f"❌ 模型关系检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("数据库模型测试")
    print("="*60)
    
    tests = [
        ("模型导入", test_models_import),
        ("模型结构", test_model_structure),
        ("元数据", test_sqlalchemy_metadata),
        ("模型关系", test_model_relationships),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("模型测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有模型测试通过！")
        print("\n下一步:")
        print("1. 确保PostgreSQL服务正在运行")
        print("2. 运行数据库迁移创建表")
        print("3. 测试向量存储功能")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())