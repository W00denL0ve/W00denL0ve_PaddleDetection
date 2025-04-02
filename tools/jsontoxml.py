import os
import json
import xml.etree.ElementTree as ET
import shutil

def create_pascal_voc(filename, width, height, objects, save_path):
    """创建VOC格式的XML标注文件"""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "pingpong"
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = save_path
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    
    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = "pinpang"
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])
    
    tree = ET.ElementTree(annotation)
    tree.write(save_path)

def process_game_folder(game_dir, output_dir, game_name):
    """处理单个游戏文件夹的数据"""
    # 设置输出目录
    output_img_dir = os.path.join(output_dir, "JPEGImages")
    output_ann_dir = os.path.join(output_dir, "Annotations")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)
    
    # 读取annotations.json
    json_path = os.path.join(game_dir, "annotations.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} 不存在")
        return
        
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    frames_dir = os.path.join(game_dir, "frames")
    box_size = 32  # 边界框大小
    img_width = 1920  # 图片宽度
    img_height = 1080  # 图片高度
    
    for frame_id, data in annotations.items():
        # 获取球的位置
        x, y = data["ball_position"]["x"], data["ball_position"]["y"]
        if x == -1 or y == -1:  # 跳过无效帧
            continue
            
        # 计算边界框坐标
        xmin = max(0, x - box_size // 2)
        ymin = max(0, y - box_size // 2)
        xmax = min(img_width, x + box_size // 2)
        ymax = min(img_height, y + box_size // 2)
        
        # 构建文件名
        frame_name = f"frame_{int(frame_id):06d}"
        new_img_name = f"{game_name}_{frame_name}.png"
        xml_name = f"{game_name}_{frame_name}.xml"
        
        # 源文件和目标文件路径
        src_img = os.path.join(frames_dir, f"{frame_name}.png")
        dst_img = os.path.join(output_img_dir, new_img_name)
        xml_path = os.path.join(output_ann_dir, xml_name)
        
        # 如果源图片存在，复制图片并创建XML
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            objects = [{
                "name": "pinpang",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            }]
            create_pascal_voc(new_img_name, img_width, img_height, objects, xml_path)

def jsontoxml():
    # 设置基础路径
    base_dir = r"D:\dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    # 处理训练集
    print("处理训练集...")
    for game_folder in [d for d in os.listdir(train_dir) if d.startswith("game_")]:
        game_path = os.path.join(train_dir, game_folder)
        if os.path.isdir(game_path):
            print(f"处理 {game_folder}...")
            process_game_folder(game_path, train_dir, game_folder)
    
    # 处理验证集
    print("\n处理验证集...")
    for game_folder in [d for d in os.listdir(val_dir) if d.startswith("game_")]:
        game_path = os.path.join(val_dir, game_folder)
        if os.path.isdir(game_path):
            print(f"处理 {game_folder}...")
            process_game_folder(game_path, val_dir, game_folder)
    
    print("\n数据处理完成！")

if __name__ == "__main__":
    jsontoxml()