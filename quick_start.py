"""
БЫСТРЫЙ СТАРТ: Полная проверка и запуск обучения
"""
import subprocess
import sys
from pathlib import Path

def run_command(description, command):
    """Запустить команду и вернуть результат"""
    print(f"\n{'='*70}")
    print(f"📌 {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Ошибка при выполнении: {description}")
        return False
    
    return True

def main():
    print("="*70)
    print("🚀 ПОЛНЫЙ ЦИКЛ: ПРОВЕРКА → ОБУЧЕНИЕ → ТЕСТИРОВАНИЕ")
    print("="*70)
    
    # 1. Проверка системы
    if not run_command("1️⃣ Проверка системы", "python check_system.py"):
        print("\n❌ Проверка системы не прошла!")
        print("Исправьте ошибки и запустите снова.")
        sys.exit(1)
    
    # 2. Запрос на начало обучения
    print("\n" + "="*70)
    print("2️⃣ Готовы начать обучение?")
    print("="*70)
    print("\n⏰ Время обучения:")
    print("   - На GPU: ~30-45 минут")
    print("   - На CPU: несколько часов")
    
    response = input("\n🚀 Начать обучение? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n❌ Обучение отменено.")
        sys.exit(0)
    
    # 3. Обучение
    if not run_command("3️⃣ Обучение модели YOLO11", "python train_model.py"):
        print("\n❌ Обучение завершилось с ошибкой!")
        sys.exit(1)
    
    # 4. Успешное завершение
    print("\n" + "="*70)
    print("🎉 ВСЁ ГОТОВО!")
    print("="*70)
    
    print("\n📍 Обученная модель: models/best.pt")
    print("\n💡 Следующие шаги:")
    print("   1. Добавьте студентов в базу: python face_database.py")
    print("   2. Протестируйте модель: python detect_violations.py --source 0")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
