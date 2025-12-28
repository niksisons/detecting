"""
Утилиты для работы с отчетами
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import config


def load_report(report_path: str) -> List[Dict]:
    """Загрузить отчет из JSON"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def report_to_excel(report_path: str, output_path: str = None):
    """Конвертировать отчет в Excel"""
    data = load_report(report_path)
    df = pd.DataFrame(data)
    
    if output_path is None:
        output_path = Path(report_path).with_suffix('.xlsx')
    
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"✅ Excel отчет сохранен: {output_path}")
    return output_path


def generate_statistics_charts(report_path: str, output_dir: str = None):
    """Генерация графиков статистики"""
    data = load_report(report_path)
    df = pd.DataFrame(data)
    
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "charts"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Стиль графиков
    sns.set_style("whitegrid")
    
    # 1. График по типам нарушений
    plt.figure(figsize=(10, 6))
    violation_counts = df['type'].value_counts()
    colors = [config.COLORS.get(vtype, (128, 128, 128)) for vtype in violation_counts.index]
    colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]  # BGR -> RGB
    
    plt.bar(violation_counts.index, violation_counts.values, color=colors_rgb)
    plt.title('Распределение нарушений по типам', fontsize=14, fontweight='bold')
    plt.xlabel('Тип нарушения')
    plt.ylabel('Количество')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'violations_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. График по нарушителям
    plt.figure(figsize=(12, 6))
    person_counts = df['person_name'].value_counts().head(10)  # Топ-10
    plt.barh(person_counts.index, person_counts.values, color='coral')
    plt.title('Топ-10 нарушителей', fontsize=14, fontweight='bold')
    plt.xlabel('Количество нарушений')
    plt.ylabel('Студент')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_violators.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. График длительности нарушений
    plt.figure(figsize=(10, 6))
    df['duration_seconds'].hist(bins=20, color='skyblue', edgecolor='black')
    plt.title('Распределение длительности нарушений', fontsize=14, fontweight='bold')
    plt.xlabel('Длительность (секунды)')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig(output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. График по времени (если есть данные о времени)
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['hour'] = df['start_time'].dt.hour
        
        plt.figure(figsize=(12, 6))
        hour_counts = df['hour'].value_counts().sort_index()
        plt.plot(hour_counts.index, hour_counts.values, marker='o', linewidth=2, markersize=8)
        plt.title('Распределение нарушений по часам', fontsize=14, fontweight='bold')
        plt.xlabel('Час дня')
        plt.ylabel('Количество нарушений')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'violations_by_hour.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Тепловая карта: тип нарушения vs студент
    if len(df) > 0:
        pivot_table = df.pivot_table(
            values='id', 
            index='person_name', 
            columns='type', 
            aggfunc='count', 
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
        plt.title('Тепловая карта нарушений', fontsize=14, fontweight='bold')
        plt.xlabel('Тип нарушения')
        plt.ylabel('Студент')
        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Графики сохранены в: {output_dir}")
    return output_dir


def generate_summary_report(report_path: str):
    """Генерация текстового резюме"""
    data = load_report(report_path)
    df = pd.DataFrame(data)
    
    summary = []
    summary.append("=" * 60)
    summary.append("📊 СВОДНЫЙ ОТЧЕТ ПО НАРУШЕНИЯМ ДИСЦИПЛИНЫ")
    summary.append("=" * 60)
    summary.append("")
    
    # Общая статистика
    summary.append("📈 ОБЩАЯ СТАТИСТИКА:")
    summary.append(f"   Всего нарушений: {len(df)}")
    summary.append(f"   Уникальных нарушителей: {df['person_name'].nunique()}")
    summary.append(f"   Средняя длительность: {df['duration_seconds'].mean():.1f} сек")
    summary.append(f"   Общее время нарушений: {df['duration_seconds'].sum():.1f} сек ({df['duration_seconds'].sum()/60:.1f} мин)")
    summary.append("")
    
    # По типам
    summary.append("📋 ПО ТИПАМ НАРУШЕНИЙ:")
    for vtype, count in df['type'].value_counts().items():
        percentage = (count / len(df)) * 100
        avg_duration = df[df['type'] == vtype]['duration_seconds'].mean()
        summary.append(f"   {vtype}:")
        summary.append(f"      Количество: {count} ({percentage:.1f}%)")
        summary.append(f"      Средняя длительность: {avg_duration:.1f} сек")
    summary.append("")
    
    # Топ нарушителей
    summary.append("👥 ТОП-5 НАРУШИТЕЛЕЙ:")
    for i, (person, count) in enumerate(df['person_name'].value_counts().head(5).items(), 1):
        person_data = df[df['person_name'] == person]
        most_common_violation = person_data['type'].mode()[0]
        summary.append(f"   {i}. {person}: {count} нарушений")
        summary.append(f"      Чаще всего: {most_common_violation}")
    summary.append("")
    
    # Временной анализ
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        summary.append("🕐 ВРЕМЕННОЙ АНАЛИЗ:")
        summary.append(f"   Период: {df['start_time'].min()} - {df['start_time'].max()}")
        
        df['hour'] = df['start_time'].dt.hour
        peak_hour = df['hour'].mode()[0]
        summary.append(f"   Пиковый час: {peak_hour}:00")
    
    summary.append("")
    summary.append("=" * 60)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Сохранение в файл
    summary_path = Path(report_path).with_suffix('.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n✅ Резюме сохранено: {summary_path}")
    return summary_text


def main():
    """Демонстрация работы с отчетами"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Работа с отчетами о нарушениях")
    parser.add_argument("report", type=str, help="Путь к JSON отчету")
    parser.add_argument("--excel", action="store_true", help="Конвертировать в Excel")
    parser.add_argument("--charts", action="store_true", help="Создать графики")
    parser.add_argument("--summary", action="store_true", help="Создать резюме")
    parser.add_argument("--all", action="store_true", help="Выполнить все операции")
    
    args = parser.parse_args()
    
    report_path = args.report
    
    if not Path(report_path).exists():
        print(f"❌ Отчет не найден: {report_path}")
        return
    
    if args.all:
        args.excel = args.charts = args.summary = True
    
    if args.excel:
        report_to_excel(report_path)
    
    if args.charts:
        generate_statistics_charts(report_path)
    
    if args.summary:
        generate_summary_report(report_path)
    
    if not (args.excel or args.charts or args.summary):
        print("Используйте --help для справки")


if __name__ == "__main__":
    main()
