"""
Проверка GPU и готовности к обучению
"""
import torch

print("=" * 60)
print("🎮 ПРОВЕРКА ВИДЕОКАРТЫ ДЛЯ ОБУЧЕНИЯ")
print("=" * 60)

# Проверка PyTorch версии
print(f"\n📦 PyTorch версия: {torch.__version__}")

# Проверка CUDA
print(f"\n🔥 CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA версия: {torch.version.cuda}")
    print(f"   cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"   Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"      Название: {torch.cuda.get_device_name(i)}")
        print(f"      Память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"      Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Тестирование GPU
    print(f"\n🧪 Тест производительности:")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU работает корректно!")
        
        # Очистка памяти
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Ошибка при тестировании GPU: {e}")
    
    print(f"\n💾 Используемая память GPU:")
    print(f"   Выделено: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   Зарезервировано: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("\n⚠️  CUDA недоступна!")
    print("   Возможные причины:")
    print("   1. Не установлены драйверы NVIDIA")
    print("   2. Установлена CPU версия PyTorch")
    print("   3. GPU не поддерживает CUDA")
    print("\n   Для установки CUDA версии PyTorch:")
    print("   pip uninstall torch torchvision -y")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)
print("✨ Проверка завершена!")
print("=" * 60)
