import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import json

def filter_dna_sequences(sequences, valid_chars='AGCT'):
    """
    过滤序列，只保留包含指定字符的序列
    
    参数:
    sequences (list): 输入DNA序列列表
    valid_chars (str): 有效的DNA字符
    
    返回:
    list: 过滤后的序列列表
    """
    filtered_sequences = []
    
    for seq in sequences:
        # 将序列转换为大写
        seq = seq.upper().replace('\n', '')
        
        # 检查序列是否只包含有效字符
        if all(char in valid_chars for char in seq):
            filtered_sequences.append(seq)
    
    return filtered_sequences

def get_sequences(gen_seqs, cell_type, level):
    """
    Get sequences for a specific cell type and level
    
    Parameters:
    cell_type (str): Cell type name
    level (int): Category level (0-4)
    
    Returns:
    list: List of sequences
    """
    if cell_type in gen_seqs and f'level_{level}' in gen_seqs[cell_type]:
        return gen_seqs[cell_type][f'level_{level}']
    else:
        return []

# Updated function to get all sequences for a cell type
def get_cell_type_sequences(cell_type):
    """
    Get all sequences for a specific cell type across all levels
    
    Parameters:
    cell_type (str): Cell type name
    
    Returns:
    dict: Dictionary with level as key and sequences as value
    """
    if cell_type in gen_seqs:
        return gen_seqs[cell_type]
    else:
        return {}

# Updated function to get statistics
def get_sequence_stats(gen_seqs, utr5_cells):
    """
    Get comprehensive statistics about loaded sequences
    
    Returns:
    dict: Statistics dictionary
    """
    stats = {
        'total_sequences': 0,
        'by_level': {},
        'by_cell_type': {},
    }
    
    # Initialize counters
    for cell_type in utr5_cells:
        stats['by_cell_type'][cell_type] = 0
    
    for level in range(5):
        stats['by_level'][f'level_{level}'] = 0
    
    # Count sequences
    for cell_type in utr5_cells:
        if cell_type in gen_seqs:
            for level in range(5):
                level_key = f'level_{level}'
                if level_key in gen_seqs[cell_type]:
                    sequences = gen_seqs[cell_type][level_key]
                    count = len(sequences)
                    
                    stats['total_sequences'] += count
                    stats['by_level'][level_key] += count
                    stats['by_cell_type'][cell_type] += count
    
    return stats


def analyze_sequence_overlap_comprehensive(gen_seqs_dict, filter_mode='none'):
    """
    全面分析所有celltype和level之间的序列重叠情况，并可选择过滤重叠序列
    
    参数:
    gen_seqs_dict (dict): 序列字典 {celltype: {level_0: [...], level_1: [...]}}
    filter_mode (str): 过滤模式
        - 'none': 不过滤，仅分析
        - 'within_celltype': 只过滤每个celltype内level间的重叠
        - 'global': 过滤所有celltype和level间的重叠
    
    返回:
    tuple: (filtered_gen_seqs, analysis_results, fig)
        - filtered_gen_seqs: 过滤后的序列字典
        - analysis_results: 分析结果字典
        - fig: 可视化图表
    """
    print(f"开始全面序列重叠分析，过滤模式: {filter_mode}")
    
    # 收集所有序列和其来源信息
    all_sequences = {}  # {sequence: [(celltype, level), ...]}
    celltype_level_seqs = {}  # {(celltype, level): set(sequences)}
    
    for celltype, levels in gen_seqs_dict.items():
        for level_key, sequences in levels.items():
            try:
                level = int(level_key.split('_')[1])
            except:
                level = level_key
            key = (celltype, level)
            
            # 清理序列
            clean_seqs = set()
            for seq in sequences:
                clean_seq = seq.replace('\n', '').upper()
                if clean_seq:  # 确保序列非空
                    clean_seqs.add(clean_seq)
                    
                    # 记录序列的来源
                    if clean_seq not in all_sequences:
                        all_sequences[clean_seq] = []
                    all_sequences[clean_seq].append(key)
            
            celltype_level_seqs[key] = clean_seqs
    
    print(f"总共收集到 {len(all_sequences)} 个唯一序列")
    print(f"涉及 {len(celltype_level_seqs)} 个 celltype-level 组合")
    
    # 分析重叠情况
    analysis_results = analyze_overlaps(all_sequences, celltype_level_seqs)
    
    # 根据过滤模式处理序列
    if filter_mode == 'none':
        filtered_gen_seqs = gen_seqs_dict.copy()
        print("不进行过滤，保持原始序列")
    else:
        filtered_gen_seqs = filter_overlapping_sequences(
            gen_seqs_dict, all_sequences, filter_mode
        )
    
    # 创建可视化
    fig = create_overlap_visualization(analysis_results, celltype_level_seqs)
    
    return filtered_gen_seqs, analysis_results, fig

def analyze_overlaps(all_sequences, celltype_level_seqs):
    """
    分析序列重叠的详细情况
    """
    print("分析重叠情况...")
    
    # 找到重叠序列
    overlapping_seqs = {seq: sources for seq, sources in all_sequences.items() if len(sources) > 1}
    unique_seqs = {seq: sources for seq, sources in all_sequences.items() if len(sources) == 1}
    
    print(f"重叠序列数量: {len(overlapping_seqs)}")
    print(f"唯一序列数量: {len(unique_seqs)}")
    
    # 分析celltype内level间重叠
    within_celltype_overlaps = defaultdict(list)
    cross_celltype_overlaps = []
    
    for seq, sources in overlapping_seqs.items():
        # 按celltype分组
        celltype_groups = defaultdict(list)
        for celltype, level in sources:
            celltype_groups[celltype].append(level)
        
        # 检查是否存在celltype内的重叠
        for celltype, levels in celltype_groups.items():
            if len(levels) > 1:
                within_celltype_overlaps[celltype].append((seq, levels))
        
        # 检查是否存在跨celltype的重叠
        if len(celltype_groups) > 1:
            cross_celltype_overlaps.append((seq, sources))
    
    # 计算统计信息
    analysis_results = {
        'total_sequences': len(all_sequences),
        'unique_sequences': len(unique_seqs),
        'overlapping_sequences': len(overlapping_seqs),
        'within_celltype_overlaps': dict(within_celltype_overlaps),
        'cross_celltype_overlaps': cross_celltype_overlaps,
        'celltype_level_seqs': celltype_level_seqs,
        'overlap_matrix': create_overlap_matrix(celltype_level_seqs)
    }
    
    # 打印详细统计
    print_overlap_statistics(analysis_results)
    
    return analysis_results

def create_overlap_matrix(celltype_level_seqs):
    """
    创建celltype-level间的重叠矩阵
    """
    keys = list(celltype_level_seqs.keys())
    n = len(keys)
    overlap_matrix = np.zeros((n, n))
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i == j:
                overlap_matrix[i, j] = len(celltype_level_seqs[key1])
            else:
                overlap = len(celltype_level_seqs[key1].intersection(celltype_level_seqs[key2]))
                overlap_matrix[i, j] = overlap
    
    return {
        'matrix': overlap_matrix,
        'labels': [f"{ct}_L{lv}" for ct, lv in keys]
    }

def filter_overlapping_sequences(gen_seqs_dict, all_sequences, filter_mode):
    """
    根据指定模式过滤重叠序列
    """
    print(f"开始过滤重叠序列，模式: {filter_mode}")
    
    filtered_gen_seqs = {}
    sequences_to_remove = set()
    
    if filter_mode == 'within_celltype':
        # 只过滤每个celltype内level间的重叠
        for seq, sources in all_sequences.items():
            celltype_groups = defaultdict(list)
            for celltype, level in sources:
                celltype_groups[celltype].append(level)
            
            # 如果同一celltype有多个level包含此序列，标记为需要移除
            for celltype, levels in celltype_groups.items():
                if len(levels) > 1:
                    sequences_to_remove.add(seq)
                    break
    
    elif filter_mode == 'global':
        # 过滤所有重叠（任意两个celltype-level间不能有重叠）
        for seq, sources in all_sequences.items():
            if len(sources) > 1:
                sequences_to_remove.add(seq)
    
    print(f"标记需要移除的序列数量: {len(sequences_to_remove)}")
    
    # 执行过滤
    for celltype, levels in gen_seqs_dict.items():
        filtered_gen_seqs[celltype] = {}
        for level_key, sequences in levels.items():
            filtered_sequences = []
            removed_count = 0
            
            for seq in sequences:
                clean_seq = seq.replace('\n', '').upper()
                if clean_seq not in sequences_to_remove:
                    filtered_sequences.append(seq)  # 保持原始格式
                else:
                    removed_count += 1
            
            filtered_gen_seqs[celltype][level_key] = filtered_sequences
            
            if removed_count > 0:
                print(f"  {celltype} {level_key}: 移除 {removed_count} 个重叠序列，保留 {len(filtered_sequences)} 个")
    
    return filtered_gen_seqs

def print_overlap_statistics(analysis_results):
    """
    打印详细的重叠统计信息
    """
    print("\n" + "="*60)
    print("重叠分析统计")
    print("="*60)
    
    print(f"总序列数: {analysis_results['total_sequences']}")
    print(f"唯一序列数: {analysis_results['unique_sequences']}")
    print(f"重叠序列数: {analysis_results['overlapping_sequences']}")
    print(f"重叠率: {analysis_results['overlapping_sequences']/analysis_results['total_sequences']*100:.2f}%")
    
    print(f"\n=== Celltype内Level间重叠 ===")
    within_overlaps = analysis_results['within_celltype_overlaps']
    if within_overlaps:
        for celltype, overlaps in within_overlaps.items():
            print(f"{celltype}: {len(overlaps)} 个重叠序列")
            # 显示前3个例子
            for i, (seq, levels) in enumerate(overlaps[:3]):
                print(f"  例子 {i+1}: Levels {levels} - {seq[:30]}...")
            if len(overlaps) > 3:
                print(f"  ... 还有 {len(overlaps)-3} 个")
    else:
        print("没有发现celltype内level间重叠")
    
    print(f"\n=== 跨Celltype重叠 ===")
    cross_overlaps = analysis_results['cross_celltype_overlaps']
    if cross_overlaps:
        print(f"跨celltype重叠序列数: {len(cross_overlaps)}")
        # 显示前3个例子
        for i, (seq, sources) in enumerate(cross_overlaps[:3]):
            sources_str = ", ".join([f"{ct}_L{lv}" for ct, lv in sources])
            print(f"  例子 {i+1}: {sources_str} - {seq[:30]}...")
        if len(cross_overlaps) > 3:
            print(f"  ... 还有 {len(cross_overlaps)-3} 个")
    else:
        print("没有发现跨celltype重叠")

def create_overlap_visualization(analysis_results, celltype_level_seqs):
    """
    创建重叠分析的可视化图表
    """
    fig = plt.figure(figsize=(30, 16))
    
    # 1. 重叠矩阵热图
    ax1 = plt.subplot(2, 3, (1, 2))
    overlap_data = analysis_results['overlap_matrix']
    
    # 创建热图
    mask = np.triu(np.ones_like(overlap_data['matrix'], dtype=bool), k=1)
    sns.heatmap(overlap_data['matrix'], 
                mask=mask,
                annot=False, 
                fmt='.0f', 
                cmap='YlOrRd',
                xticklabels=overlap_data['labels'], 
                yticklabels=overlap_data['labels'],
                ax=ax1,
                square=True)
    ax1.set_title('Sequence Overlap Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 2. 每个celltype-level的序列数量
    ax2 = plt.subplot(2, 3, 3)
    counts = [len(seqs) for seqs in celltype_level_seqs.values()]
    labels = [f"{ct}_L{lv}" for ct, lv in celltype_level_seqs.keys()]
    
    bars = ax2.bar(range(len(counts)), counts, alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Number of Sequences')
    ax2.set_title('Sequences per CellType-Level', fontsize=14, fontweight='bold')
    
    # 3. 重叠统计饼图
    ax3 = plt.subplot(2, 3, 4)
    overlap_stats = [
        analysis_results['unique_sequences'],
        analysis_results['overlapping_sequences']
    ]
    labels_pie = ['Unique Sequences', 'Overlapping Sequences']
    colors = ['lightblue', 'lightcoral']
    
    ax3.pie(overlap_stats, labels=labels_pie, autopct='%1.1f%%', colors=colors)
    ax3.set_title('Sequence Overlap Distribution', fontsize=14, fontweight='bold')
    
    # 4. Celltype内重叠统计
    ax4 = plt.subplot(2, 3, 5)
    within_overlaps = analysis_results['within_celltype_overlaps']
    
    if within_overlaps:
        celltypes = list(within_overlaps.keys())
        overlap_counts = [len(overlaps) for overlaps in within_overlaps.values()]
        
        bars = ax4.bar(celltypes, overlap_counts, alpha=0.7, color='orange')
        ax4.set_ylabel('Number of Overlapping Sequences')
        ax4.set_title('Within-CellType Level Overlaps', fontsize=14, fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, overlap_counts):
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No Within-CellType\nOverlaps Found', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Within-CellType Level Overlaps', fontsize=14, fontweight='bold')
    
    # 5. 跨celltype重叠分析
    ax5 = plt.subplot(2, 3, 6)
    cross_overlaps = analysis_results['cross_celltype_overlaps']
    
    if cross_overlaps:
        # 统计涉及的celltype数量
        celltype_involvement = defaultdict(int)
        for seq, sources in cross_overlaps:
            involved_celltypes = set(ct for ct, lv in sources)
            for ct in involved_celltypes:
                celltype_involvement[ct] += 1
        
        if celltype_involvement:
            celltypes = list(celltype_involvement.keys())
            involvement_counts = list(celltype_involvement.values())
            
            bars = ax5.bar(celltypes, involvement_counts, alpha=0.7, color='red')
            ax5.set_ylabel('Number of Cross-Overlapping Sequences')
            ax5.set_title('Cross-CellType Overlaps by CellType', fontsize=14, fontweight='bold')
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
            
            # 添加数值标签
            for bar, count in zip(bars, involvement_counts):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom')
    else:
        ax5.text(0.5, 0.5, 'No Cross-CellType\nOverlaps Found', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Cross-CellType Overlaps', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Comprehensive Sequence Overlap Analysis', fontsize=16, y=0.98)
    
    return fig

def compare_before_after_filtering(original_gen_seqs, filtered_gen_seqs):
    """
    比较过滤前后的序列统计
    """
    print("\n" + "="*60)
    print("过滤前后对比")
    print("="*60)
    
    orig_stats = {}
    filt_stats = {}
    
    # 统计原始数据
    for celltype, levels in original_gen_seqs.items():
        orig_stats[celltype] = {}
        for level_key, sequences in levels.items():
            orig_stats[celltype][level_key] = len(sequences)
    
    # 统计过滤后数据
    for celltype, levels in filtered_gen_seqs.items():
        filt_stats[celltype] = {}
        for level_key, sequences in levels.items():
            filt_stats[celltype][level_key] = len(sequences)
    
    # 打印对比
    print(f"{'CellType':<15} {'Level':<10} {'Original':<10} {'Filtered':<10} {'Removed':<10} {'Removal %':<12}")
    print("-" * 75)
    
    total_orig = 0
    total_filt = 0
    
    for celltype in original_gen_seqs.keys():
        for level in range(5):
            level_key = f'level_{level}'
            orig_count = orig_stats.get(celltype, {}).get(level_key, 0)
            filt_count = filt_stats.get(celltype, {}).get(level_key, 0)
            removed = orig_count - filt_count
            removal_pct = (removed / orig_count * 100) if orig_count > 0 else 0
            
            print(f"{celltype:<15} {level_key:<10} {orig_count:<10} {filt_count:<10} {removed:<10} {removal_pct:<12.1f}")
            
            total_orig += orig_count
            total_filt += filt_count
    
    total_removed = total_orig - total_filt
    total_removal_pct = (total_removed / total_orig * 100) if total_orig > 0 else 0
    
    print("-" * 75)
    print(f"{'TOTAL':<15} {'ALL':<10} {total_orig:<10} {total_filt:<10} {total_removed:<10} {total_removal_pct:<12.1f}")


############### Visuliaztion
def select_best_sequences_for_celltype(pivoted_results_df, celltype, timo_prediction_stats, timo_training_stats):
    """
    为指定的细胞类型选择具有最佳表达差异的序列
    
    参数:
    pivoted_results_df (DataFrame): 已经pivot过的DataFrame
    celltype (str): 目标细胞类型基础名称(如'Colo320_Rep1')
    
    返回:
    DataFrame: 原始DataFrame加上表达优势分析列，并按表达优势排序
    """
    
    # 检查可用的细胞类型
    pred_diff_columns = [col for col in pivoted_results_df.columns if col.endswith('_pred_diff')]
    available_celltypes = [col.split('_pred_diff')[0] for col in pred_diff_columns]
    print(f"可用的细胞类型: {available_celltypes}")
    
    if celltype not in available_celltypes:
        print(f"错误: 找不到细胞类型 '{celltype}'")
        return pd.DataFrame()
    
    print(f"找到目标细胞类型: {celltype}")
    
    # 创建结果DataFrame的副本
    result_df = pivoted_results_df.copy()
    
    # 使用特定celltype的预测列
    pred_diff_col = f"{celltype}_pred_diff"
    pred_mass_col = f"{celltype}_pred_center_of_mass"
    
    # 找出非目标细胞类型
    other_celltypes = [ct for ct in available_celltypes if ct != celltype]
    other_pred_diff_columns = [f"{ct}_pred_diff" for ct in other_celltypes]
    
    print(f"\n=== Other Cell Diff 计算过程 ===")
    print(f"目标细胞类型: {celltype}")
    print(f"其他细胞类型: {other_celltypes}")
    print(f"其他细胞类型的预测差异列: {other_pred_diff_columns}")
    
    # 计算非目标细胞类型的最大预测差异
    if other_celltypes and other_pred_diff_columns:
        # 步骤1: 找到每个序列在其他细胞类型中的最大预测差异值
        result_df['max_other_pred_diff'] = result_df[other_pred_diff_columns].max(axis=1)
        
        # 步骤2: 找出具有最大预测差异的非目标细胞类型
        def find_max_celltype(row):
            max_value = row['max_other_pred_diff']
            for ct in other_celltypes:
                col = f"{ct}_pred_diff"
                if col in row and row[col] == max_value:
                    return ct
            return "N/A"
        
        result_df['max_other_celltype'] = result_df.apply(find_max_celltype, axis=1)
        
        # 步骤3: 获取最大其他细胞类型的质量中心
        def get_max_celltype_mass(row):
            max_celltype = row['max_other_celltype']
            if max_celltype != "N/A":
                col = f"{max_celltype}_pred_center_of_mass"
                if col in row:
                    return row[col]
            return None
        
        result_df['max_other_pred_center_of_mass'] = result_df.apply(get_max_celltype_mass, axis=1)
        
        # 打印示例来说明计算过程
        print(f"\n--- 示例序列的 Other Cell Diff 计算 ---")
        example_seq = result_df.iloc[0]
        print(f"序列: {example_seq['sequence'][:30]}...")
        print(f"目标细胞类型 {celltype} 的预测差异: {example_seq[pred_diff_col]:.4f}")
        
        for ct in other_celltypes:
            col = f"{ct}_pred_diff"
            if col in example_seq:
                print(f"其他细胞类型 {ct} 的预测差异: {example_seq[col]:.4f}")
        
        print(f"其他细胞类型中的最大预测差异: {example_seq['max_other_pred_diff']:.4f}")
        print(f"具有最大预测差异的其他细胞类型: {example_seq['max_other_celltype']}")
        
    else:
        result_df['max_other_pred_diff'] = float('-inf')
        result_df['max_other_celltype'] = "N/A"
        result_df['max_other_pred_center_of_mass'] = None
    
    # 检查统计数据是否存在
    if celltype not in timo_training_stats or celltype not in timo_prediction_stats:
        print(f"警告: 找不到细胞类型 '{celltype}' 的统计数据")
        print(f"训练统计中可用的细胞类型: {list(timo_training_stats.keys())}")
        print(f"预测统计中可用的细胞类型: {list(timo_prediction_stats.keys())}")
        # 使用默认值继续处理
        training_stats = {'diff_max': 0, 'diff_mean': 0, 'mass_max': 2.5, 'mass_mean': 2.5}
        prediction_stats = {'diff_max': 0, 'diff_mean': 0, 'mass_max': 2.5, 'mass_mean': 2.5}
    else:
        training_stats = timo_training_stats[celltype]
        prediction_stats = timo_prediction_stats[celltype]
    
    # 获取训练数据和预测数据的统计信息
    training_diff_max = training_stats['diff_max']
    training_diff_mean = training_stats['diff_mean']
    training_mass_max = training_stats['mass_max']
    training_mass_mean = training_stats['mass_mean']
    
    prediction_diff_max = prediction_stats['diff_max']
    prediction_diff_mean = prediction_stats['diff_mean']
    prediction_mass_max = prediction_stats['mass_max']
    prediction_mass_mean = prediction_stats['mass_mean']
    
    # 计算特定细胞类型相对于训练数据的优势
    # 1. 与训练数据最大值的差距（越+越好）
    result_df['training_diff_max_advantage'] = result_df[pred_diff_col] - training_diff_max
    if pred_mass_col in result_df.columns:
        result_df['training_mass_max_advantage'] = result_df[pred_mass_col] - training_mass_max
    
    # 2. 与训练数据平均值的差距（越大越好）
    result_df['training_diff_mean_advantage'] = result_df[pred_diff_col] - training_diff_mean
    if pred_mass_col in result_df.columns:
        result_df['training_mass_mean_advantage'] = result_df[pred_mass_col] - training_mass_mean

    # 3. 与预测数据最大值的差距（越+越好）
    result_df['prediction_diff_max_advantage'] = result_df[pred_diff_col] - prediction_diff_max
    if pred_mass_col in result_df.columns:
        result_df['prediction_mass_max_advantage'] = result_df[pred_mass_col] - prediction_mass_max
    
    # 4. 与预测数据平均值的差距（越大越好）
    result_df['prediction_diff_mean_advantage'] = result_df[pred_diff_col] - prediction_diff_mean
    if pred_mass_col in result_df.columns:
        result_df['prediction_mass_mean_advantage'] = result_df[pred_mass_col] - prediction_mass_mean

    # 计算对其他细胞类型的优势
    # 1. 预测差异优势 - 这就是 other_cell_diff_advantage 的计算方式！
    result_df['other_cell_diff_advantage'] = result_df[pred_diff_col] - result_df['max_other_pred_diff']
    
    print(f"\n--- Other Cell Diff Advantage 计算公式 ---")
    print(f"other_cell_diff_advantage = {celltype}_pred_diff - max_other_pred_diff")
    print(f"即: 目标细胞类型的预测差异 - 其他细胞类型中的最大预测差异")
    print(f"")
    print(f"示例计算:")
    example_seq = result_df.iloc[0]
    target_diff = example_seq[pred_diff_col]
    max_other_diff = example_seq['max_other_pred_diff']
    other_advantage = example_seq['other_cell_diff_advantage']
    print(f"{target_diff:.4f} - {max_other_diff:.4f} = {other_advantage:.4f}")
    print(f"")
    print(f"解释:")
    print(f"- 如果 other_cell_diff_advantage > 0: 说明序列在目标细胞类型中的表现比在其他任何细胞类型中都好")
    print(f"- 如果 other_cell_diff_advantage < 0: 说明序列在某个其他细胞类型中的表现更好")
    print(f"- 值越大，说明该序列对目标细胞类型的特异性越强")
    
    # 2. 质量中心优势
    if pred_mass_col in result_df.columns and 'max_other_pred_center_of_mass' in result_df.columns:
        result_df['other_cell_mass_advantage'] = result_df[pred_mass_col] - result_df['max_other_pred_center_of_mass']
    
    # 主要按与训练最大值的差距排序（越大越好，即超过训练数据最大值越多越好）
    result_df = result_df.sort_values('training_diff_max_advantage', ascending=False)
    
    # 打印一些统计信息
    print(f"\n=== {celltype} 优势分析统计 ===")
    print(f"训练数据 - diff_max: {training_diff_max:.4f}, diff_mean: {training_diff_mean:.4f}")
    print(f"预测数据 - diff_max: {prediction_diff_max:.4f}, diff_mean: {prediction_diff_mean:.4f}")
    
    if 'training_diff_max_advantage' in result_df.columns:
        positive_training_adv = (result_df['training_diff_max_advantage'] > 0).sum()
        print(f"超过训练最大值的序列数量: {positive_training_adv}/{len(result_df)} ({positive_training_adv/len(result_df)*100:.1f}%)")
    
    if 'prediction_diff_max_advantage' in result_df.columns:
        positive_prediction_adv = (result_df['prediction_diff_max_advantage'] > 0).sum()
        print(f"超过预测最大值的序列数量: {positive_prediction_adv}/{len(result_df)} ({positive_prediction_adv/len(result_df)*100:.1f}%)")
    
    if 'other_cell_diff_advantage' in result_df.columns:
        positive_other_adv = (result_df['other_cell_diff_advantage'] > 0).sum()
        print(f"对目标细胞类型更特异的序列数量: {positive_other_adv}/{len(result_df)} ({positive_other_adv/len(result_df)*100:.1f}%)")
    
    return result_df

def generate_and_select_sequences(gen_seqs_dict, predictor, celltype, level, utrdata, timo_prediction_stats, timo_training_stats):
    """
    主函数：从指定细胞类型和级别生成序列、过滤并选择最佳序列
    
    参数:
    gen_seqs_dict (dict): 生成序列字典 {celltype: {level_0: [...], level_1: [...]}}
    predictor: Parade预测器
    celltype (str): 目标细胞类型(如'Colo320_Rep1')
    level (int): 目标级别 (0-4)
    utrdata: UTR数据
    
    返回:
    DataFrame: 包含最佳序列的DataFrame
    """
    # 检查细胞类型是否存在
    if celltype not in gen_seqs_dict:
        print(f"错误: 找不到细胞类型 '{celltype}'")
        available_cells = list(gen_seqs_dict.keys())
        print(f"可用的细胞类型: {available_cells}")
        return pd.DataFrame()
    
    # 检查级别是否存在
    level_key = f'level_{level}'
    if level_key not in gen_seqs_dict[celltype]:
        print(f"错误: 找不到级别 '{level_key}' 在细胞类型 '{celltype}' 中")
        available_levels = list(gen_seqs_dict[celltype].keys())
        print(f"可用的级别: {available_levels}")
        return pd.DataFrame()
    
    # 获取指定细胞类型和级别的序列
    gen_seqs = gen_seqs_dict[celltype][level_key]
    
    if not gen_seqs:
        print(f"警告: {celltype} - {level_key} 中没有序列")
        return pd.DataFrame()
    
    print(f"从 {celltype} - {level_key} 加载了 {len(gen_seqs)} 个序列")
    
    # 打印前几个序列用于调试
    print("示例序列:")
    for i, seq in enumerate(gen_seqs[:3]):
        print(f"  序列 {i+1}: {seq[:50]}...")

    # 获取预测结果
    print("正在获取预测结果...")
    results = predictor.predict_and_summarize(gen_seqs, utrdata)
    results = results.pivot(index='sequence', columns='cell_type', values=['pred_diff', 'pred_center_of_mass'])
    results.columns = [f'{col[1]}_{col[0]}' for col in results.columns]
    results = results.reset_index()
    
    # 检查预测结果中的细胞类型并找到最匹配的
    pred_diff_columns = [col for col in results.columns if col.endswith('_pred_diff')]
    available_celltypes = [col.split('_pred_diff')[0] for col in pred_diff_columns]
    print(f"预测结果中可用的细胞类型: {available_celltypes}")
    
    # 尝试找到匹配的细胞类型
    target_celltype = None
    
    # 1. 首先尝试完全匹配
    if celltype in available_celltypes:
        target_celltype = celltype
        print(f"找到完全匹配的细胞类型: {target_celltype}")
    else:
        # 2. 尝试基础名称匹配（去掉_Rep后缀）
        base_celltype = celltype.split('_')[0]  # 例如 'Colo320_Rep1' -> 'Colo320'
        
        # 查找包含基础名称的细胞类型
        matching_celltypes = [ct for ct in available_celltypes if base_celltype in ct]
        
        if matching_celltypes:
            target_celltype = matching_celltypes[0]  # 使用第一个匹配的
            print(f"找到基础名称匹配的细胞类型: {target_celltype} (基于 {base_celltype})")
        else:
            print(f"错误: 无法找到匹配 '{celltype}' 或 '{base_celltype}' 的细胞类型")
            print(f"可用的细胞类型: {available_celltypes}")
            return pd.DataFrame()
    
    # 选择最佳序列
    best_sequences = select_best_sequences_for_celltype(results, target_celltype, timo_prediction_stats, timo_training_stats)
    
    # 添加来源信息
    if not best_sequences.empty:
        best_sequences['source_celltype'] = celltype
        best_sequences['source_level'] = level
        best_sequences['target_celltype_used'] = target_celltype
    
    return best_sequences

def batch_generate_and_select_sequences(gen_seqs_dict, predictor, target_celltype, utrdata, timo_prediction_stats, timo_training_stats, levels=None):
    """
    批量处理：为指定细胞类型的所有级别生成和选择序列
    
    参数:
    gen_seqs_dict (dict): 生成序列字典
    predictor: Parade预测器
    target_celltype (str): 目标细胞类型完整名称 (如'Colo320_Rep1')
    utrdata: UTR数据
    levels (list): 要处理的级别列表，默认为[0,1,2,3,4]
    
    返回:
    dict: {level: DataFrame} 每个级别的最佳序列结果
    """
    if levels is None:
        levels = [0, 1, 2, 3, 4]
    
    # 直接使用完整的细胞类型名称
    if target_celltype not in gen_seqs_dict:
        print(f"错误: 找不到细胞类型 '{target_celltype}'")
        available_cells = list(gen_seqs_dict.keys())
        print(f"可用的细胞类型: {available_cells}")
        return {}
    
    matching_celltypes = [target_celltype]
    print(f"找到匹配的细胞类型: {matching_celltypes}")
    
    results_by_level = {}
    
    # 为每个级别处理每个匹配的细胞类型
    for level in levels:
        print(f"\n=== 处理级别 {level} ===")
        level_results = []
        
        for celltype in matching_celltypes:
            print(f"\n处理 {celltype} - Level {level}")
            result = generate_and_select_sequences(
                gen_seqs_dict, predictor, celltype, level, utrdata, timo_prediction_stats, timo_training_stats
            )
            
            if not result.empty:
                level_results.append(result)
            else:
                print(f"  {celltype} - Level {level}: 没有有效结果")
        
        # 合并同一级别的所有结果
        if level_results:
            combined_result = pd.concat(level_results, ignore_index=True)
            # 按优势重新排序
            if 'training_diff_max_advantage' in combined_result.columns:
                combined_result = combined_result.sort_values('training_diff_max_advantage', ascending=False)
            results_by_level[level] = combined_result
            print(f"  级别 {level} 总共获得 {len(combined_result)} 个序列")
        else:
            print(f"  级别 {level}: 没有有效结果")
    
    return results_by_level

def compare_levels_for_celltype(results_by_level, target_celltype):
    """
    比较同一细胞类型不同级别的结果
    
    参数:
    results_by_level (dict): batch_generate_and_select_sequences的输出
    target_celltype (str): 目标细胞类型完整名称
    
    返回:
    DataFrame: 级别比较摘要
    """
    comparison_data = []
    
    for level, df in results_by_level.items():
        if df.empty:
            continue
            
        # 计算统计信息
        target_diff_col = f"{target_celltype}_pred_diff"
        target_mass_col = f"{target_celltype}_pred_center_of_mass"
        
        stats = {
            'level': level,
            'total_sequences': len(df),
            'unique_source_celltypes': df['source_celltype'].nunique() if 'source_celltype' in df.columns else 0
        }
        
        # 添加目标细胞类型的预测统计
        if target_diff_col in df.columns:
            stats.update({
                'pred_diff_mean': df[target_diff_col].mean(),
                'pred_diff_std': df[target_diff_col].std(),
                'pred_diff_max': df[target_diff_col].max(),
                'pred_diff_min': df[target_diff_col].min()
            })
        
        if target_mass_col in df.columns:
            stats.update({
                'pred_mass_mean': df[target_mass_col].mean(),
                'pred_mass_std': df[target_mass_col].std(),
                'pred_mass_max': df[target_mass_col].max(),
                'pred_mass_min': df[target_mass_col].min()
            })
        
        # 添加优势统计
        advantage_cols = [col for col in df.columns if 'advantage' in col]
        for adv_col in advantage_cols:
            if adv_col in df.columns:
                stats[f'{adv_col}_mean'] = df[adv_col].mean()
                stats[f'{adv_col}_positive_pct'] = (df[adv_col] > 0).sum() / len(df) * 100
        
        comparison_data.append(stats)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df
    
def visualize_level_comparison(results_by_level, target_celltype, timo_prediction_stats, timo_training_stats):
    """
    可视化不同级别的比较结果，分为两个画布：
    1. 第一个画布：前2个图（分布图）
    2. 第二个画布：每个Level一个散点图
    
    参数:
    results_by_level (dict): 级别结果字典
    target_celltype (str): 目标细胞类型
    
    返回:
    tuple: (fig1, fig2) 两个matplotlib.figure.Figure对象
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # 获取预测数据基准
    prediction_stats = timo_prediction_stats.get(target_celltype, {})
    
    # 准备数据
    plot_data = []
    scatter_data = []  # 为散点图准备数据
    
    for level, df in results_by_level.items():
        if df.empty:
            continue
            
        target_diff_col = f"{target_celltype}_pred_diff"
        target_mass_col = f"{target_celltype}_pred_center_of_mass"
        
        # 添加diff数据
        if target_diff_col in df.columns:
            for val in df[target_diff_col]:
                plot_data.append({
                    'level': f'Level {level}',
                    'metric': 'Prediction Difference',
                    'value': val
                })
        
        # 添加mass数据
        if target_mass_col in df.columns:
            for val in df[target_mass_col]:
                plot_data.append({
                    'level': f'Level {level}',
                    'metric': 'Center of Mass',
                    'value': val
                })
        
        # 为散点图准备数据 - 整合center of mass和diff
        if target_diff_col in df.columns and target_mass_col in df.columns:
            for _, row in df.iterrows():
                scatter_data.append({
                    'level': level,
                    'center_of_mass': row[target_mass_col],
                    'diff': row[target_diff_col]
                })
    
    if not plot_data:
        print("没有数据可供可视化")
        return None, None
    
    plot_df = pd.DataFrame(plot_data)
    scatter_df = pd.DataFrame(scatter_data)
    
    # ================== 第一个画布：前2个图 ==================
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Diff值的小提琴图 - 带预测基准线
    ax1 = axes1[0]
    diff_data = plot_df[plot_df['metric'] == 'Prediction Difference']
    if not diff_data.empty:
        sns.violinplot(data=diff_data, x='level', y='value', ax=ax1, alpha=0.7)
        ax1.set_title(f'Prediction Difference Distribution by Level ({target_celltype})', fontsize=16)
        ax1.set_ylabel('Prediction Difference', fontsize=14)
        ax1.set_xlabel('Level', fontsize=14)
        
        # 添加基准线
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Line')
        
        if 'diff_max' in prediction_stats:
            ax1.axhline(y=prediction_stats['diff_max'], color='red', linestyle='--', 
                       linewidth=2, label=f'Prediction Max ({prediction_stats["diff_max"]:.3f})')
        
        if 'diff_mean' in prediction_stats:
            ax1.axhline(y=prediction_stats['diff_mean'], color='red', linestyle=':', 
                       linewidth=2, label=f'Prediction Mean ({prediction_stats["diff_mean"]:.3f})')
        
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 2. Mass值的小提琴图 - 带预测基准线
    ax2 = axes1[1]
    mass_data = plot_df[plot_df['metric'] == 'Center of Mass']
    if not mass_data.empty:
        sns.violinplot(data=mass_data, x='level', y='value', ax=ax2, alpha=0.7)
        ax2.set_title(f'Center of Mass Distribution by Level ({target_celltype})', fontsize=16)
        ax2.set_ylabel('Center of Mass', fontsize=14)
        ax2.set_xlabel('Level', fontsize=14)
        
        # 添加基准线
        if 'mass_max' in prediction_stats:
            ax2.axhline(y=prediction_stats['mass_max'], color='red', linestyle='--', 
                       linewidth=2, label=f'Prediction Max ({prediction_stats["mass_max"]:.3f})')
        
        if 'mass_mean' in prediction_stats:
            ax2.axhline(y=prediction_stats['mass_mean'], color='red', linestyle=':', 
                       linewidth=2, label=f'Prediction Mean ({prediction_stats["mass_mean"]:.3f})')
        
        ax2.legend(fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig1.suptitle(f'Level Comparison Analysis - Distribution Overview', 
                  fontsize=20, y=1.02)
    
    # ================== 第二个画布：每个Level的散点图（同一行） ==================
    if not scatter_df.empty:
        levels = sorted(scatter_df['level'].unique())
        n_levels = len(levels)
        
        # 计算统一的坐标轴范围
        all_mass = scatter_df['center_of_mass']
        all_diff = scatter_df['diff']
        
        mass_min, mass_max = all_mass.min(), all_mass.max()
        diff_min, diff_max = all_diff.min(), all_diff.max()
        
        # 给坐标轴留一些边距
        mass_range = mass_max - mass_min
        diff_range = diff_max - diff_min
        
        mass_lim = (mass_min - 0.05 * mass_range, mass_max + 0.05 * mass_range)
        diff_lim = (diff_min - 0.05 * diff_range, diff_max + 0.05 * diff_range)
        
        # 所有散点图排成一行
        fig2, axes2 = plt.subplots(1, n_levels, figsize=(5*n_levels, 5))
        
        # 如果只有一个子图，axes2不是数组
        if n_levels == 1:
            axes2 = [axes2]
        
        # 定义颜色映射
        level_colors = {}
        for level in levels:
            if level == 3:
                level_colors[level] = 'red'  # Level 3 设为红色
            elif level == 0:
                level_colors[level] = 'blue'
            elif level == 1:
                level_colors[level] = 'green'
            elif level == 2:
                level_colors[level] = 'orange'
            elif level == 4:
                level_colors[level] = 'purple'
            else:
                level_colors[level] = plt.cm.Set1(level / max(levels))
        
        # 为每个level创建散点图
        for i, level in enumerate(levels):
            ax = axes2[i]
            level_data = scatter_df[scatter_df['level'] == level]
            
            # 绘制散点
            ax.scatter(level_data['center_of_mass'], level_data['diff'], 
                      c=level_colors[level], alpha=0.6, s=8, edgecolors='none')
            
            # 添加主要分界线
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax.axvline(x=2.5, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # 添加高性能区域参考线 (center_of_mass > 3 & diff > 0.5)
            ax.axhline(y=0.5, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2)
            ax.axvline(x=3.0, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2)
            
            # 设置统一的坐标轴范围
            ax.set_xlim(mass_lim)
            ax.set_ylim(diff_lim)
            
            # 设置标签和标题
            ax.set_xlabel('Center of Mass', fontsize=14)
            ax.set_ylabel('Prediction Difference', fontsize=14)
            ax.set_title(f'Level {level} (n={len(level_data)})', fontsize=16, 
                        color=level_colors[level], fontweight='bold')
            
            # 计算四个象限的统计信息（保持原来的2.5和0分界）
            total = len(level_data)
            if total > 0:
                # 四个象限（原来的分界）
                q1 = ((level_data['diff'] > 0) & (level_data['center_of_mass'] < 2.5)).sum()  # 左上：High Diff, Low Mass
                q2 = ((level_data['diff'] > 0) & (level_data['center_of_mass'] >= 2.5)).sum() # 右上：High Diff, High Mass
                q3 = ((level_data['diff'] <= 0) & (level_data['center_of_mass'] < 2.5)).sum() # 左下：Low Diff, Low Mass
                q4 = ((level_data['diff'] <= 0) & (level_data['center_of_mass'] >= 2.5)).sum() # 右下：Low Diff, High Mass
                
                # 计算高性能区域的统计（center_of_mass > 3 & diff > 0.5）
                high_perf = ((level_data['center_of_mass'] > 3.0) & (level_data['diff'] > 0.5)).sum()
                
                # 计算百分比
                q1_pct = q1 / total * 100
                q2_pct = q2 / total * 100
                q3_pct = q3 / total * 100
                q4_pct = q4 / total * 100
                high_perf_pct = high_perf / total * 100
                
                # 添加四个象限的统计标注（保持原来的位置）
                # 左上角 (High Diff, Low Mass)
                ax.text(0.02, 0.98, f'{q1}\n({q1_pct:.1f}%)', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='top', alpha=0.8,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                
                # 右上角 (High Diff, High Mass) - 理想象限
                ax.text(0.98, 0.98, f'{q2}\n({q2_pct:.1f}%)', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='top', horizontalalignment='right', alpha=0.8,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                
                # 左下角 (Low Diff, Low Mass)
                ax.text(0.02, 0.02, f'{q3}\n({q3_pct:.1f}%)', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='bottom', alpha=0.8,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
                # 右下角 (Low Diff, High Mass)
                ax.text(0.98, 0.02, f'{q4}\n({q4_pct:.1f}%)', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='bottom', horizontalalignment='right', alpha=0.8,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                
                # 添加高性能区域统计（右上角区域内，稍微偏下一点）
                ax.text(0.98, 0.85, f'Elite:\n{high_perf} ({high_perf_pct:.1f}%)', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='top', horizontalalignment='right', alpha=0.9,
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8, edgecolor='darkgreen', linewidth=2))
            
            # 设置网格
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴字体大小
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        fig2.suptitle(f'Center of Mass vs Prediction Difference by Level - {target_celltype}', 
                      fontsize=18, y=1.02)
        
    else:
        fig2 = None
    
    return fig1, fig2

def run_single_cell_line_analysis(cell_type, gen_seqs, predictor, utrdata, timo_prediction_stats, timo_training_stats):
    
    # 批量处理所有级别
    results_by_level = batch_generate_and_select_sequences(gen_seqs, predictor, cell_type, utrdata, timo_prediction_stats, timo_training_stats)
    
    # 比较不同级别
    comparison = compare_levels_for_celltype(results_by_level, cell_type)
    print(f"\n级别比较结果:")
    print(comparison)
    
    # 可视化
    fig = visualize_level_comparison(results_by_level, cell_type, timo_prediction_stats, timo_training_stats)
    
    # 获取Level 4的最佳结果，按pred_diff排序
    if 4 in results_by_level:
        level_4_results = results_by_level[4].sort_values(f'{cell_type}_pred_diff', ascending=False)
        print(f"\nLevel 4 最佳序列 (按 {cell_type}_pred_diff 排序):")
        print(level_4_results[['sequence', f'{cell_type}_pred_diff', 
                              'training_diff_max_advantage', 'prediction_diff_max_advantage']].head())
    
    return results_by_level, comparison, fig