import os
import sys

def train_model():
    # 设置基础路径
    base_dir = r"D:\programfiles\codes\Py_files\PaddleDetection-release-2.8.1"
    work_dir = r"D:\dataset\work"
    
    # 创建工作目录
    vdl_dir = os.path.join(work_dir, "vdl_dir")
    model_dir = os.path.join(work_dir, "model")
    os.makedirs(vdl_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 切换到PaddleDetection目录并设置环境变量
    os.chdir(base_dir)
    os.environ['PYTHONPATH'] = base_dir
    
    # 构建训练命令
    train_cmd = [
        "python", "tools/train.py",
        "-c", "configs/picodet/ppq.yml",
        "--use_vdl=true",
        f"--vdl_log_dir={vdl_dir}",
        "--eval",
        "-o", f"save_dir={model_dir}"
    ]
    
    # 执行训练
    print("开始训练模型...")
    os.system(" ".join(train_cmd))

if __name__ == "__main__":
    train_model()
