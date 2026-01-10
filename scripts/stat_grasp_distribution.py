import sys
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 假设原有的 tools 路径和导入依然有效
sys.path.append(".")
sys.path.append("..")
from tools.utils import makelogger
from tools.utils import contact_ids as CONTACT_IDS


class GrabGraspStat:
    def __init__(self, data_dir, logger=None):
        self.data_dir = data_dir
        if logger is None:
            log_dir = os.path.join(self.data_dir, "stat.log")
            self.logger = makelogger(log_dir=log_dir, mode="a").info
        else:
            self.logger = logger

        self.all_grasps = glob.glob(self.data_dir + "/*/*.npy")
        self.contact_id_to_name = {val: key for key, val in CONTACT_IDS.items()}

    def analyze(self):
        grasp_data_list = []

        for grasp_file in tqdm(self.all_grasps, desc="Analyzing grasps"):
            try:
                data = np.load(grasp_file, allow_pickle=True).item()

                # Get object scale
                diagonal_scale = data["object"]["scale"]

                # Get grasp type
                contacts = data["object"]["contact"]
                r_hand = any(self.contact_id_to_name[c].startswith("R_") for c in contacts if c > 0)
                l_hand = any(self.contact_id_to_name[c].startswith("L_") for c in contacts if c > 0)
                g_type = "both" if (r_hand and l_hand) else "right" if r_hand else "left" if l_hand else "none"

                grasp_data_list.append({"scale": diagonal_scale, "type": g_type})

                # if diagonal_scale > 0.28 and g_type != "both":
                #     self.logger(f"Large object with single-hand grasped: {grasp_file}, scale: {diagonal_scale:.3f} m")

            except Exception:
                continue

        # Visualize grasp type distribution over object scale intervals
        df = pd.DataFrame(grasp_data_list)
        # 1. 动态生成 Bin 区间
        bin_size = 0.02
        max_scale = df["scale"].max()
        # 创建从 0 到 max_scale + bin_size 的范围
        bins = np.arange(0, max_scale + bin_size, bin_size)
        # 2. 使用 pd.cut 生成区间标签，格式为 [0.0, 0.02)
        df["scale_bin"] = pd.cut(df["scale"], bins=bins, right=False, precision=2, include_lowest=True)
        # 3. 统计表格
        pivot_table = df.groupby(["scale_bin", "type"]).size().unstack(fill_value=0)
        # 打印统计
        print("\n--- Grasp Type Distribution by Object Scale Intervals ---")
        print(pivot_table)
        self.visualize(pivot_table)

    def visualize(self, pivot_table):
        sns.set(style="whitegrid")

        # 绘制堆叠柱状图
        # ax 为绘图对象，方便后续微调标签
        ax = pivot_table.plot(
            kind="bar", stacked=True, figsize=(14, 7), color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]
        )

        plt.title("Grasp Type Distribution vs Object Scale Interval", fontsize=15)
        plt.xlabel("Object Scale Interval [Start, End) / meters", fontsize=12)
        plt.ylabel("Number of Grasps", fontsize=12)

        # 优化 X 轴标签：pd.cut 生成的标签默认是 Interval 对象，转为字符串更美观
        xticklabels = [f"{str(label)}" for label in pivot_table.index]
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

        plt.legend(title="Grasp Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # 保存并显示
        plt.show()


if __name__ == "__main__":
    grab_data_path = "/data/dataset/GRAB/segment_grasps"
    stat = GrabGraspStat(grab_data_path)
    stat.analyze()
