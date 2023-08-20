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

x = ['15', '30', '50', '100', '150', '200', '250', '300', '500', '1000']
y3 = [66.11, 66.52, 66.39, 67.68, 63.61, 64.53, 65.23, 65.28, 65.75, 65.71]
y4 = [65.75, 65.74, 65.65, 68.5, 65.55, 70.03, 67.67, 67.46, 68.02, 66.94]
y_range = [55, 80]
x_label = 'Embedding Dim'
title = 'abl_model_size'

# x = ['15',	'30',	'50',	'100',	'150',	'200',	'250',	'300',	'500',	'1000']
y1 = [62.97166667, 64.78833333, 65.05833333, 68.495, 69.18666667, 69.94, 70.17333333, 69.85333333, 69.72166667, 67.575]
y2 = [63.84333333,	65.245,	66.43,	69.29,	69.96666667,	70.65,	70.755,	71.15333333,	70.78166667,	68.49833333]
# y_range = [60, 72]
# x_label = 'Embedding Dim'
# title = 'model_size'

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

l1, = plt.plot(x, y1, '-o', ms=10, label='Full Fine-tune', linewidth=4, color='#1f77b4')
l2, = plt.plot(x, y2, '-o', ms=10, label='AdapterGNN', linewidth=4, color='#ff7f0e')
l3, = plt.plot(x, y3, '--', ms=10, label='Full Fine-tune', linewidth=4, color='#1f77b4')
l4, = plt.plot(x, y4, '--', ms=10, label='AdapterGNN', linewidth=4, color='#ff7f0e')

plt.xlabel(x_label, fontsize=30)
plt.xticks(x, fontsize=fontsize, rotation=50)
plt.yticks(fontsize=fontsize)
plt.ylabel('ROC-AUC(%)', fontsize=30)
first_legend = plt.legend(fontsize=20, loc="upper left", handles=[l1, l2], title='Average', title_fontsize=20)
ax = plt.gca().add_artist(first_legend)
plt.legend(fontsize=20, loc="lower right", handles=[l3, l4], title='BBBP', title_fontsize=20)

plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/工作/研究/AdapterGNN/AAAI 2024/{}.pdf'.format(title), format='pdf', dpi=300, pad_inches=0)
