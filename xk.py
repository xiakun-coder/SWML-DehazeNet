import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False
})

models = [
    'DehazeNet', 'AOD-Net', 'RI-SCNN(large)',
    'TaylorFormer(s)', 'BDT-Net', 'WLD-Net', 'SWML-Net(Ours)'
]
params = [0.009, 0.002, 2.559, 2.680, 5.24, 0.385, 0.414]
psnr = [19.82, 20.51, 29.02, 22.07, 26.84, 26.82, 29.22]

colors = ['#9467bd', '#e377c2', '#17becf', '#8c564b', '#1f77b4', '#ff7f0e', '#d62728']
markers = ['o'] * 6 + ['*']
sizes = [300] * 6 + [500]

fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

ax.set_xlim(-0.2, 6)
ax.set_ylim(19, 30)
ax.grid(True, linestyle='--', alpha=0.5)

ax.set_xlabel('Parameters (M)', weight='bold')
ax.set_ylabel('PSNR (dB)', weight='bold')

ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)

ax.tick_params(width=2.5, length=6)

for i in range(len(models)):
    ax.scatter(params[i], psnr[i],
               color=colors[i],
               marker=markers[i],
               s=sizes[i],
               edgecolors='black',
               linewidth=2.5)

    ox, oy = 0.1, 0.2
    if i == 2: ox = -0.4
    if i == 3: ox = -0.4
    if i == 4: ox = -0.5
    if i == 6: oy = 0.4

    ax.text(params[i] + ox, psnr[i] + oy, models[i],
            fontsize=16,
            weight='normal')

plt.tight_layout()
plt.savefig('FINAL_BOLD_BORDER.png', dpi=300, bbox_inches='tight')
plt.show()