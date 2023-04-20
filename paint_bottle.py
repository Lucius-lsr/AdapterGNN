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

x = ['0', '1', '2', '5', '10', '15', '30', '60', '100', '150', ]
y1 = [69.55] * 10
y2 = [66.51166667, 67.29, 68.28833333, 69.68, 69.77, 71.15333333, 70.27166667, 69.33166667, 70.07833333, 69.24333333]
y_range = [66, 72]
x_label = 'Bottleneck Dim'
title = 'abl_3'

plt.rc('font', family="Times New Roman")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(12, 6))

plt.ylim(ymax=y_range[1], ymin=y_range[0])

fontsize = 25
plt.grid(linestyle='-.', linewidth=1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.plot(x, y1, '-', ms=10, label='Full Fine-tune', linewidth=4, color='dimgray')
plt.plot(x, y2, '-o', ms=10, label='AdapterGNN', linewidth=4, color='#ff7f0e')

plt.xlabel(x_label, fontsize=30)
plt.xticks(x, fontsize=fontsize)
plt.ylabel('ROC-AUC(%)', fontsize=30)
plt.legend(fontsize=30, loc="lower right")
plt.yticks(fontsize=fontsize)

plt.yticks(fontsize=fontsize)
# plt.title(title, fontsize=30)

plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/工作/研究/AdapterGNN/{}.pdf'.format(title), format='pdf', dpi=300, pad_inches=0)
