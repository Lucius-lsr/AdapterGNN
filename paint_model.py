# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/1 16:16
@Author  : Lucius
@FileName: ssl_epoch.py
@Software: PyCharm
"""

import matplotlib
import matplotlib.pyplot as plt

# x = ['20',
#      '30',
#      '40',
#      '50',
#      '60',
#      '70',
#      '80',
#      '90',
#      '100']
# y1 = [53.6,
#       53.37,
#       53.9,
#       56.1,
#       55.63,
#       57.82,
#       60.32,
#       60.13,
#       60.81]
# y2 = [53.66,
#       54.04,
#       55.38,
#       58.47,
#       58.45,
#       59.9,
#       59.91,
#       60.87,
#       61.26]
# y3 = [55.66,
#       54.62,
#       57.46,
#       58.37,
#       58.66,
#       59.55,
#       60.53,
#       61.23,
#       61.05]
# y_range = [52, 62]
# x_label = 'Available Training Data (%)'
# title = 'abl_2_1'

x = [
     '0.5',
     '0.6',
     '0.7',
     '0.8',
     '0.9',
     '1',
     '2',
     '3',
     '4',
     '5',
     '10',
     '20',
     '30',
     '40',
     '50',
     '60',
     '70',
     '80',
     '90',
     '100']
y1 = [
      58.62,
      59.03,
      62.21,
      61.9,
      61.19,
      63.34,
      65.65,
      66.83,
      66.79,
      67.75,
      70.19,
      71.3,
      72.34,
      73.79,
      74.69,
      75.77,
      76.36,
      77.02,
      76.76,
      76.76]
y2 = [
      59.62,
      59.96,
      62.94,
      62.31,
      61.49,
      63.93,
      65.08,
      66.92,
      65.8,
      67.22,
      69.91,
      71.09,
      72.11,
      73.94,
      74.38,
      75.25,
      76.03,
      75.62,
      76.23,
      76.58]
y3 = [
      60.34,
      60.21,
      63.23,
      62.2,
      61.67,
      65.1,
      65.43,
      66.97,
      66.38,
      67.12,
      69.89,
      70.59,
      72.34,
      72.76,
      73.82,
      74.33,
      74.9,
      75.41,
      75.66,
      75.79]
y_range = [55, 80]
x_label = 'Available Training Data (%)'
title = 'abl_2_2'

x, y1, y2, y3 = x[:6], y1[:6], y2[:6], y3[:6]
title = 'abl_2_2_'
y_range = [56, 66]
#
# x, y1, y2, y3 = x[-6:], y1[-6:], y2[-6:], y3[-6:]
# title = 'abl_2_2_2'
# y_range = [74, 78]

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

plt.plot(x, y1, '-o', ms=10, label='Full Fine-tune', linewidth=4)
plt.plot(x, y2, '-o', ms=10, label='AdapterGNN-15', linewidth=4)
plt.plot(x, y3, '-o', ms=10, label='AdapterGNN-5', linewidth=4)


plt.xticks(x, fontsize=fontsize, rotation=50)
# plt.yticks([74, 78], fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.legend(fontsize=20, loc="lower right")
plt.xlabel(x_label, fontsize=30)
plt.ylabel('ROC-AUC(%)', fontsize=30)


plt.tight_layout()
# plt.show()
plt.savefig('/Users/lishengrui/Desktop/工作/研究/AdapterGNN/AAAI 2024/{}.pdf'.format(title), format='pdf', dpi=300, pad_inches=0)
