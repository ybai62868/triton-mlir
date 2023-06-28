import numpy as np
import matplotlib.pyplot as plt

# RTX 3090性能参数
peak_performance = 35.6 * 1e12  # TFLOPs (FP32)
peak_bandwidth = 936 * 1e9  # GB/s

# 设置横轴（算术密度）
arithmetic_intensities = np.logspace(-3, 2, 100)

# 计算Roofline模型的两个部分
compute_bound = peak_performance * arithmetic_intensities
memory_bound = peak_bandwidth * np.ones_like(arithmetic_intensities)

# 绘制Roofline模型
plt.figure(figsize=(10, 6))
plt.loglog(arithmetic_intensities, compute_bound, label="Compute-bound", lw=2)
plt.loglog(arithmetic_intensities, memory_bound, label="Memory-bound", lw=2)

# 设置图表样式
plt.xlabel("Arithmetic Intensity (FLOPs/byte)", fontsize=14)
plt.ylabel("Performance (FLOPs/s)", fontsize=14)
plt.title("Roofline Model - NVIDIA GeForce RTX 3090", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.gca().set_ylim(bottom=1e9)

# 显示图表
plt.show()
