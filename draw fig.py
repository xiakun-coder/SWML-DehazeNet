import matplotlib.pyplot as plt
import numpy as np

# ---------------- Data ----------------
categories = ['Channel Unscaled', 'Channel Doubled']
psnr = [20.790, 21.434]
params = np.array([385834, 6878730]) / 1e6  # Million

x = np.arange(len(categories))

# ---------------- Style ----------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.linewidth": 1.2
})

fig, ax1 = plt.subplots(figsize=(3.8, 3.2))

# ---------------- Bar: PSNR ----------------
bar_width = 0.4
bars = ax1.bar(x, psnr, width=bar_width,
               color='lightgray', edgecolor='black',
               label='PSNR (dB)')

ax1.set_ylabel("PSNR (dB)")
ax1.set_ylim(20, 22)
ax1.set_xticks(x)
ax1.set_xticklabels(categories)

# value labels (small & clean)
for i, v in enumerate(psnr):
    ax1.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

# ---------------- Line: Params ----------------
ax2 = ax1.twinx()
ax2.plot(x, params, marker='o', linestyle='--',
         color='black', linewidth=1.5,
         label='Params (M)')

ax2.set_ylabel("Parameters (M)")

for i, v in enumerate(params):
    ax2.text(i, v + 0.15, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

# ---------------- Legend ----------------
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', frameon=False)

# ---------------- Grid ----------------
ax1.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)

# ---------------- Layout ----------------
plt.tight_layout()

# ---------------- Save ----------------
plt.savefig("channel_strategy_comparison.pdf", bbox_inches='tight')
plt.savefig("channel_strategy_comparison.png", dpi=300, bbox_inches='tight')
plt.show()