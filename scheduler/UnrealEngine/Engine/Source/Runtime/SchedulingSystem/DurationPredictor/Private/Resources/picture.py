import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 设置全局字体为 Calibri
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 确保中文字符也使用 Calibri
plt.rcParams['axes.unicode_minus'] = False

# 获取所有的 txt 文件
files = glob.glob('output_*.txt')

fig = plt.figure(figsize=(20, 10))

names = ["add", "cpy", "mul", "mul_mat", "rms_norm", "rope", "silu", "soft_max"]

# 对每个文件进行处理
for k, file in enumerate(files):
    with open(file, 'r') as f:
        lines = f.readlines()
    # 提取每行的最后一个数字
    numbers = [float(line.strip().split(',')[-1]) for line in lines]
    # 计算子图的位置
    nrows = len(files) // 4
    nrows += len(files) % 4
    ax = fig.add_subplot(nrows, 4, k+1)
    
    # 画出分布图
    sns.kdeplot(numbers, fill=True, ax=ax)
    
    # 设置标题和坐标轴标签，确保使用 Calibri 字体
    ax.set_title(names[k], fontsize=40, fontfamily='Calibri', fontweight='bold')
    ax.set_xlabel('duration(ms)', fontsize=40, fontfamily='Calibri', fontweight='bold')
    ax.set_ylabel('Probability density', fontsize=40, fontfamily='Calibri', fontweight='bold')
    
    # 设置刻度标签的字体
    ax.tick_params(axis='both', which='major', labelsize=40)
    
    # 应用字体到刻度标签
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Calibri')
    
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 0.2)
    ax.get_xticklabels()[0].set_visible(False)

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()