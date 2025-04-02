import os
import shutil

def copy_config_files():
    print("开始复制配置文件...")
    # 定义基础路径
    base_dir = r"D:\programfiles\codes\Py_files\PaddleDetection-release-2.8.1"
    
    # 源文件路径
    ppq_source = os.path.join(base_dir, "configs", "ppq.yml")
    voc_ppq_source = os.path.join(base_dir, "configs", "voc_ppq.yml")
    
    # 目标路径
    picodet_dir = os.path.join(base_dir, "configs", "picodet")
    datasets_dir = os.path.join(base_dir, "configs", "datasets")
    
    # 确保目标目录存在
    os.makedirs(picodet_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    
    # 复制文件
    try:
        shutil.copy2(ppq_source, picodet_dir)
        shutil.copy2(voc_ppq_source, datasets_dir)
        print("配置文件复制成功！")
    except Exception as e:
        print(f"复制过程中出现错误: {e}")

if __name__ == "__main__":
    copy_config_files()
