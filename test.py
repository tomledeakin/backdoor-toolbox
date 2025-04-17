# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # Dùng font giống với ACM (Linux Libertine)
# rcParams['font.family'] = 'Linux Libertine O'
# # Tạo thư mục lưu kết quả (nếu chưa tồn tại)
# save_dir = "TED/"
# os.makedirs(save_dir, exist_ok=True)
#
# # Dữ liệu cho phương pháp TED (shape: 9 x 4)
# ted_data = np.array([
#     [0.9548, 0.9724, 0.9680, 0.8380],
#     [0.9772, 0.9844, 0.9920, 0.3676],
#     [0.8360, 0.8080, 0.8636, 0.4548],
#     [0.7600, 0.6356, 0.6228, 0.6796],
#     [0.8660, 0.7556, 0.9652, 0.9180],
#     [0.7912, 0.7948, 0.6288, 0.7188],
#     [0.8988, 0.8500, 0.8148, 0.6132],
#     [0.7428, 0.7516, 0.6860, 0.8904],
#     [0.9996, 0.9716, 0.9248, 0.7536]
# ])
#
# our_data = np.array([
#     [0.9948, 0.9912, 0.9944, 0.9488],
#     [0.9980, 0.9708, 0.9208, 0.8836],
#     [0.9928, 0.9924, 0.9964, 0.9356],
#     [0.9984, 0.9864, 0.9308, 0.9616],
#     [0.9516, 0.9348, 0.9172, 0.8872],
#     [0.9996, 1.0000, 0.9424, 0.9660],
#     [0.9956, 0.9944, 0.9988, 0.9812],
#     [1.0000, 1.0000, 1.0000, 1.0000],
#     [1.0000, 0.9984, 0.9980, 0.9492]
# ])
#
# # ted_data = np.array([
# #     [0.9604, 0.9546, 0.9566, 0.9230],
# #     [0.9864, 0.9620, 0.9320, 0.1400],
# #     [0.9352, 0.9180, 0.9432, 0.7200],
# #     [0.8980, 0.8556, 0.7396, 0.3450],
# #     [0.9246, 0.9400, 0.9136, 0.8928],
# #     [0.9462, 0.9370, 0.8912, 0.3476],
# #     [0.9832, 0.9740, 0.9384, 0.9996],
# #     [0.9328, 0.8916, 0.8440, 0.5436],
# #     [0.9996, 0.9996, 0.9996, 0.9388]
# # ])
# #
# # our_data = np.array([
# #     [0.9920, 0.9748, 0.9380, 0.8912],
# #     [0.9908, 0.9760, 0.9964, 0.8538],
# #     [0.9844, 0.9874, 1.0000, 0.9394],
# #     [0.9956, 0.9870, 0.9686, 0.8716],
# #     [0.9202, 0.8990, 0.9138, 0.8384],
# #     [0.9874, 0.9810, 0.9500, 0.8894],
# #     [1.0000, 1.0000, 0.9764, 0.9800],
# #     [0.9636, 0.9392, 0.9166, 0.8364],
# #     [0.9816, 0.9476, 0.9576, 0.8042]
# # ])
#
# # Các thời điểm (từ trái sang phải): 20, 10, 5, 2
# time_points = [20, 10, 5, 2]
#
# # Tạo DataFrame cho TED và chuyển sang dạng long-form
# df_ted = pd.DataFrame(ted_data, columns=time_points)
# df_ted_melted = df_ted.melt(var_name="Time", value_name="ROC AUC")
# df_ted_melted["Time"] = df_ted_melted["Time"].astype(float)
#
# # Tạo DataFrame cho Ours và chuyển sang dạng long-form
# df_our = pd.DataFrame(our_data, columns=time_points)
# df_our_melted = df_our.melt(var_name="Time", value_name="ROC AUC")
# df_our_melted["Time"] = df_our_melted["Time"].astype(float)
#
# # Thiết lập style với nền trắng, không có grid mặc định
# sns.set(style="white", font_scale=1.2)
# plt.figure(figsize=(22, 5))
#
# # Vẽ lineplot cho Ours với marker "s" (square, màu tím)
# sns.lineplot(
#     data=df_our_melted,
#     x="Time", y="ROC AUC", errorbar="sd",
#     marker="X", color="purple", label="Ours",
#     linewidth=4, markersize=15
# )
#
# # Vẽ lineplot cho TED với marker "o" (màu cam)
# sns.lineplot(
#     data=df_ted_melted,
#     x="Time", y="ROC AUC", errorbar="sd",
#     marker="v", color="orange", label="TED",
#     linewidth=4, markersize=15
# )
#
# # Đặt xticks và yticks với fontsize lớn
# plt.xticks(time_points, fontsize=30)
# plt.yticks(fontsize=30)
#
# # Đảo ngược trục x để thời điểm 20 nằm bên trái
# plt.gca().invert_xaxis()
# # Thiết lập nhãn trục
# plt.xlabel("Number of Validation Samples", fontsize=30)
# plt.ylabel("ROC AUC", fontsize=30)
#
# # Tăng kích thước của legend
# plt.legend(fontsize=30, frameon=True)
#
# # Bật grid
# plt.gca().grid(False)
#
# # Loại bỏ viền trên và phải để tạo cảm giác clean
# sns.despine()
#
# # plot_save_path = os.path.join(save_dir, "ted_lineplot.pdf")
# #
# plot_save_path = os.path.join(save_dir, "ted_lineplot.png")
# plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
# plt.show()
#
# print(f"Plot saved to: {plot_save_path}")

import os
import numpy as np
import matplotlib.pyplot as plt

# # Define the save directory and create it if it doesn't exist
# save_dir = "TED/"
# os.makedirs(save_dir, exist_ok=True)
#
# # Define the alpha values and M values.
# alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
# M_values = [20, 15, 10, 5, 2]
#
# # Define the AUC matrix; each row corresponds to an M value (in the same order as M_values)
# auc_matrix = np.array([
#     [0.3404, 0.9792, 0.9980, 0.9912, 0.9984],
#     [0.1440, 0.9628, 0.9924, 0.9852, 0.9944],
#     [0.2598, 0.8736, 0.9656, 0.9964, 0.9544],
#     [0.2602, 0.9304, 0.9136, 1.0000, 1.0000],
#     [0.8792, 0.8792, 0.8792, 0.8792, 0.8792]
# ])
#
# # Create the heatmap with a larger figure size to accommodate bigger labels and annotations
# plt.figure(figsize=(12, 10))
# heatmap = plt.imshow(auc_matrix, cmap='bwr', aspect='auto')
#
# # Configure the axes: Increase font sizes for ticks and labels.
# plt.xticks(ticks=np.arange(len(alpha_values)), labels=alpha_values, fontsize=28)
# plt.yticks(ticks=np.arange(len(M_values)), labels=M_values, fontsize=28)
# plt.xlabel('Alpha', fontsize=28)
# plt.ylabel('M', fontsize=28)
# plt.title('Heatmap of AUC Values for Given Alpha and M', fontsize=28)
#
# # Add a color bar with increased font size for its label and ticks.
# cbar = plt.colorbar(heatmap)
# cbar.set_label('AUC', fontsize=28)
# cbar.ax.tick_params(labelsize=28)
#
# # Annotate each cell with the corresponding AUC value.
# # Increase the fontsize for the annotations inside the heatmap.
# for i in range(len(M_values)):
#     for j in range(len(alpha_values)):
#         plt.text(j, i, f"{auc_matrix[i, j]:.4f}", ha="center", va="center",
#                  color="black", fontsize=28)
#
# plt.tight_layout()
#
# # Save the heatmap as a PNG file in the specified save directory
# output_path = os.path.join(save_dir, "heatmap.png")
# plt.savefig(output_path)
# print(f"Heatmap saved to {output_path}")
#
# # Display the heatmap
# plt.show()
#


import os
import zipfile

# Xác định thư mục gốc (dựng theo vị trí của test.py)
project_dir = os.path.dirname(os.path.abspath(__file__))
zip_filename = os.path.join(project_dir, "data", "celeba-dataset.zip")
extract_dir = os.path.join(project_dir, "data", "celeba_dataset")

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Giải nén thành công!")

