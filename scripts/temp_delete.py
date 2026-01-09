from pathlib import Path


def clean_npy_files(folder_path):
    # 将路径转为 Path 对象
    root_dir = Path(folder_path)

    # 查找所有以 .npy 结尾的文件 (包括子目录)
    # rglob = recursive glob
    npy_files = list(root_dir.rglob("*.npy"))

    print(f"找到 {len(npy_files)} 个 .npy 文件，准备删除...")

    for file in npy_files:
        try:
            file.unlink()  # 执行删除
            # print(f"已删除: {file}") # 如果文件太多建议注释掉这行
        except Exception as e:
            print(f"删除失败 {file}: {e}")

    print("清理完成。")


# 使用示例
target_folder = "/data/dataset/GRAB/extract/grab"
clean_npy_files(target_folder)
