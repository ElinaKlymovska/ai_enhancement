#!/usr/bin/env python3
"""
Demo script for ADetailer + ControlNet Enhanced v2.0 features
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from config_validator import ConfigValidator


def setup_demo_logging():
    """Налаштування логування для демо"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo.log')
        ]
    )
    return logging.getLogger("Demo")


def demo_config_validation():
    """Демонстрація валідації конфігурації"""
    print("\n" + "="*60)
    print("🔍 ДЕМОНСТРАЦІЯ: Валідація конфігурації")
    print("="*60)
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Файл конфігурації не знайдено: {config_path}")
        return False
    
    try:
        validator = ConfigValidator(config_path)
        is_valid, errors, warnings = validator.validate_all()
        
        print(f"📁 Файл: {config_path}")
        print(f"✅ Валідність: {'Так' if is_valid else 'Ні'}")
        
        if errors:
            print(f"\n❌ Помилки ({len(errors)}):")
            for error in errors:
                print(f"  • {error}")
        
        if warnings:
            print(f"\n⚠️  Попередження ({len(warnings)}):")
            for warning in warnings:
                print(f"  • {warning}")
        
        if not errors and not warnings:
            print("\n🎉 Конфігурація ідеальна!")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Помилка валідації: {e}")
        return False


def demo_enhanced_config():
    """Демонстрація розширеної конфігурації"""
    print("\n" + "="*60)
    print("⚙️  ДЕМОНСТРАЦІЯ: Розширена конфігурація")
    print("="*60)
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Файл конфігурації не знайдено: {config_path}")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Показуємо нові секції
        new_sections = ['quality', 'logging']
        for section in new_sections:
            if section in config:
                print(f"\n📋 Секція '{section}':")
                section_data = config[section]
                for key, value in section_data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"\n❌ Секція '{section}' відсутня")
        
        # Показуємо покращені параметри
        if 'general' in config:
            general = config['general']
            print(f"\n🔧 Покращені параметри general:")
            for key in ['max_retries', 'retry_delay', 'timeout_seconds']:
                if key in general:
                    print(f"  {key}: {general[key]}")
                else:
                    print(f"  {key}: не встановлено")
        
    except Exception as e:
        print(f"❌ Помилка читання конфігурації: {e}")


def demo_batch_features():
    """Демонстрація нових можливостей batch обробки"""
    print("\n" + "="*60)
    print("🚀 ДЕМОНСТРАЦІЯ: Нові можливості batch обробки")
    print("="*60)
    
    print("📊 Покращене логування:")
    print("  • Детальні повідомлення з часовими мітками")
    print("  • Різні рівні логування (DEBUG, INFO, WARNING, ERROR)")
    print("  • Логування у файл з ротацією")
    
    print("\n🔄 Retry механізм:")
    print("  • Автоматичні повторні спроби при помилках")
    print("  • Експоненціальний backoff (2s, 4s, 8s)")
    print("  • Налаштовувана кількість спроб")
    
    print("\n⚡ Паралельна обробка:")
    print("  • Множинні робочі процеси")
    print("  • Прогрес-бари для кожного процесу")
    print("  • Ефективне використання ресурсів")
    
    print("\n📈 Статистика обробки:")
    print("  • Кількість успішних/невдалих зображень")
    print("  • Детальні логи для кожного зображення")
    print("  • Автоматичне завершення при помилках")


def demo_quality_control():
    """Демонстрація контролю якості"""
    print("\n" + "="*60)
    print("🎯 ДЕМОНСТРАЦІЯ: Контроль якості")
    print("="*60)
    
    print("🔍 Автоматична перевірка якості:")
    print("  • Детекція обличчя з налаштовуваною впевненістю")
    print("  • Аналіз різкості зображення")
    print("  • Перевірка рівня розмиття")
    
    print("\n📊 Налаштування порогів:")
    print("  • min_face_detection_confidence: 0.5")
    print("  • max_blur_threshold: 100")
    print("  • min_sharpness_score: 0.6")
    
    print("\n🎨 Режими обробки:")
    print("  • portrait_mode: Покращення портретів")
    print("  • group_mode: Обробка групових фото")
    print("  • Автоматичне визначення типу зображення")


def demo_gui_features():
    """Демонстрація GUI можливостей"""
    print("\n" + "="*60)
    print("🖥️  ДЕМОНСТРАЦІЯ: Графічний інтерфейс")
    print("="*60)
    
    print("📱 Вкладки налаштувань:")
    print("  • General: Backend, endpoint, retry параметри")
    print("  • B-Pass: Prompts, параметри обробки")
    print("  • I/O: Директорії та налаштування файлів")
    print("  • Quality: Параметри контролю якості")
    print("  • Logging: Налаштування логування")
    
    print("\n🔧 Функції редагування:")
    print("  • Відкриття/збереження конфігурацій")
    print("  • Валідація в реальному часі")
    print("  • Вибір директорій через діалоги")
    print("  • Автодоповнення та перевірка типів")
    
    print("\n💾 Збереження та завантаження:")
    print("  • Формат YAML з підтримкою Unicode")
    print("  • Автоматичне створення резервних копій")
    print("  • Експорт/імпорт налаштувань")


def create_sample_input_structure():
    """Створення прикладної структури директорій"""
    print("\n" + "="*60)
    print("📁 Створення прикладної структури директорій")
    print("="*60)
    
    directories = ['input', 'work', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Створено директорію: {directory}")
        else:
            print(f"📁 Директорію вже існує: {directory}")
    
    # Створення прикладового файлу README в input
    readme_path = "input/README.txt"
    if not os.path.exists(readme_path):
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("Помістіть сюди зображення для обробки\n")
            f.write("Підтримувані формати: PNG, JPG, JPEG, WEBP\n")
        print(f"✅ Створено приклад: {readme_path}")


def main():
    """Головна функція демо"""
    print("🎉 ДЕМОНСТРАЦІЯ ADetailer + ControlNet Enhanced v2.0")
    print("="*60)
    
    logger = setup_demo_logging()
    logger.info("Початок демонстрації нових можливостей")
    
    try:
        # Демонстрація всіх нових функцій
        demo_config_validation()
        demo_enhanced_config()
        demo_batch_features()
        demo_quality_control()
        demo_gui_features()
        create_sample_input_structure()
        
        print("\n" + "="*60)
        print("🎊 ДЕМОНСТРАЦІЯ ЗАВЕРШЕНА!")
        print("="*60)
        
        print("\n📋 Наступні кроки:")
        print("1. Запустіть GUI: python config_gui.py")
        print("2. Валідуйте конфігурацію: python config_validator.py config.yaml")
        print("3. Запустіть batch обробку: python batch.py --workers 2")
        print("4. Перегляньте логи: tail -f adetailer_batch.log")
        
        logger.info("Демонстрація успішно завершена")
        
    except Exception as e:
        logger.error(f"Помилка під час демонстрації: {e}")
        print(f"\n❌ Помилка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
