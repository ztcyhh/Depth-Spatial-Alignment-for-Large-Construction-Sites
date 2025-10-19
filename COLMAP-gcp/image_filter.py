import os
import shutil
from pathlib import Path
import glob
from tqdm import tqdm

def filter_and_copy_images(source_folder):
    """
    遍历源文件夹中的图片，跳过n为4倍数的图片，将其他图片复制到目标文件夹
    
    Args:
        source_folder (str): 源文件夹路径
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在！")
        return
    
    # 创建目标文件夹路径（在源文件夹同级目录下创建topath_075）
    source_path = Path(source_folder)
    target_folder = source_path.parent / "images_075"
    
    # 创建目标文件夹
    target_folder.mkdir(exist_ok=True)
    print(f"目标文件夹：{target_folder}")
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif']
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_folder, ext)))
        image_files.extend(glob.glob(os.path.join(source_folder, ext.upper())))
    
    # 按文件名排序
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 统计信息
    copied_count = 0
    skipped_count = 0
    
    # 遍历图片文件
    for i, image_path in enumerate(tqdm(image_files, desc="处理图片")):
        # 获取文件名（不含扩展名）
        filename = Path(image_path).stem
        
        # 检查文件名是否包含数字
        try:
            # 尝试从文件名中提取数字
            # 这里假设文件名格式为 "image_001.jpg" 或 "001.jpg" 等
            import re
            numbers = re.findall(r'\d+', filename)
            
            if numbers:
                # 使用最后一个数字作为n
                n = int(numbers[-1])
                
                # 如果n是4的倍数，跳过
                if n % 4 == 0:
                    print(f"跳过 {filename} (n={n} 是4的倍数)")
                    skipped_count += 1
                    continue
            
        except (ValueError, IndexError):
            # 如果无法提取数字，默认复制
            pass
        
        # 复制文件
        try:
            target_path = target_folder / Path(image_path).name
            shutil.copy2(image_path, target_path)
            copied_count += 1
            print(f"复制: {Path(image_path).name}")
            
        except Exception as e:
            print(f"复制失败 {Path(image_path).name}: {e}")
    
    print(f"\n处理完成！")
    print(f"复制了 {copied_count} 个文件")
    print(f"跳过了 {skipped_count} 个文件")
    print(f"目标文件夹：{target_folder}")

def main():
    """主函数"""
    print("图片筛选和复制工具")
    print("=" * 50)
    
    # 获取用户输入的源文件夹路径
    while True:
        source_folder = "/data/zt/project/colmap/xinyang/0418/tower4_only_75/images"
        
        # 去除可能的引号
        source_folder = source_folder.strip('"\'')
        
        if source_folder:
            break
        else:
            print("请输入有效的文件夹路径！")
    
    # 执行筛选和复制
    filter_and_copy_images(source_folder)

if __name__ == "__main__":
    main()
