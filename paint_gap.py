# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/1 16:16
@Author  : Lucius
@FileName: ssl_epoch.py
@Software: PyCharm
"""

import re

import matplotlib
import matplotlib.pyplot as plt

x_label = 'Embedding Dim'
title = 'abl_gap'
x = ['1', '5', '15', '30', '50', '100', '150', '200', '250', '300', '500', '1000']
y1 = [0.234, 0.2177, 0.201, 0.1895, 0.1807, 0.1668, 0.1542, 0.1428, 0.1307, 0.1187, 0.084, 0.0419]
y2 = [0.2977, 0.2761, 0.269, 0.2663, 0.2664, 0.2732, 0.2816, 0.2943, 0.3061, 0.328, 0.4379, 0.605]
y3 = [0.2327, 0.2204, 0.2019, 0.1913, 0.1819, 0.1744, 0.1643, 0.1604, 0.1548, 0.1498, 0.1343, 0.1081]
y4 = [0.2949, 0.2775, 0.2679, 0.2713, 0.2692, 0.2759, 0.2819, 0.2836, 0.2838, 0.2897, 0.3139, 0.3521]
y_range = [0, max(y1 + y2 + y3 + y4) * 1.3]
plt.rc('font', family="Times New Roman")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(6, 6))

plt.ylim(ymax=y_range[1], ymin=y_range[0])

fontsize = 25
plt.grid(linestyle='-.', linewidth=1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

l1, = plt.plot(x, y1, '--', ms=10, label='Full Fine-tune', linewidth=4, color='#1f77b4')
l2, = plt.plot(x, y2, '-', ms=10, label='Full Fine-tune', linewidth=4, color='#1f77b4')
l3, = plt.plot(x, y3, '--', ms=10, label='AdapterGNN', linewidth=4, color='#ff7f0e')
l4, = plt.plot(x, y4, '-', ms=10, label='AdapterGNN', linewidth=4, color='#ff7f0e')

plt.xlabel(x_label, fontsize=30)
plt.xticks(x, fontsize=fontsize, rotation=50)
plt.yticks(fontsize=fontsize)
plt.ylabel('Error', fontsize=30)
first_legend = plt.legend(fontsize=20, loc="upper left", handles=[l2, l4], title='Test', title_fontsize=20)
ax = plt.gca().add_artist(first_legend)
plt.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.31, 0.7), handles=[l1, l3], title='Training', title_fontsize=20)

plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/工作/研究/AdapterGNN/AAAI 2024/{}.pdf'.format(title), format='pdf', dpi=300, pad_inches=0)
