import os
import cv2
import numpy as np
from tqdm import tqdm

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"已创建输出目录: {output_path}")
    else:
        print(f"输出目录已存在: {output_path}")

def MA_adaptive(a_path, b_path, c_path, output_path):
    """
    根据mask图片融合A和B文件夹中的图片
    
    参数:
    a_path: encode step数量小的模型推理结果图像保存路径
    b_path: encode step数量大的模型推理结果图像保存路径
    c_path: mask的的图像保存路径（这里mask像素值为0的是CT阈值低于2800的，1的是CT阈值高于2800的）
    output_path: 输出文件夹路径
    """
    
    a_files = [f for f in os.listdir(a_path) if os.path.isfile(os.path.join(a_path, f))]
    b_files = [f for f in os.listdir(b_path) if os.path.isfile(os.path.join(b_path, f))]
    c_files = [f for f in os.listdir(c_path) if os.path.isfile(os.path.join(c_path, f))]
    
    common_files = list(set(a_files) & set(b_files) & set(c_files))
    print(f"找到 {len(common_files)} 个共同的图片文件")
    
    if not common_files:
        print("没有找到共同的图片文件，将中止")
        return
    
    for filename in tqdm(common_files, desc="进度"):
        try:
            a_img = cv2.imread(os.path.join(a_path, filename))
            b_img = cv2.imread(os.path.join(b_path, filename))
            c_mask = cv2.imread(os.path.join(c_path, filename), cv2.IMREAD_GRAYSCALE)
            
            if a_img is None or b_img is None or c_mask is None:
                print(f"警告: 无法读取图片 {filename}，已跳过")
                continue
            
            # 确保所有图片尺寸相同
            if a_img.shape[:2] != b_img.shape[:2] or a_img.shape[:2] != c_mask.shape[:2]:
                print(f"警告: 图片 {filename} 尺寸不匹配，已跳过")
                continue
            
            # 归一化mask到0和1（便于计算权重）
            mask_normalized = (c_mask / 255).astype(np.float32)
            
            a_weight = 0.8 - 0.6 * mask_normalized
            b_weight = 0.2 + 0.6 * mask_normalized
            
            a_weight = np.expand_dims(a_weight, axis=-1)
            b_weight = np.expand_dims(b_weight, axis=-1)
            
            a_img_float = a_img.astype(np.float32)
            b_img_float = b_img.astype(np.float32)
            
            fused_img = a_img_float * a_weight + b_img_float * b_weight
            
            fused_img = np.clip(fused_img, 0, 255).astype(np.uint8)
            
            output_filename = os.path.join(output_path, filename)
            cv2.imwrite(output_filename, fused_img)
            
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    folder_a = "./output/A"       # encode step数量小的模型推理结果图像保存路径
    folder_b = "./output/B"       # encode step数量大的模型推理结果图像保存路径
    folder_c = "./mask"       # mask的的图像保存路径（这里mask像素值为0的是CT阈值低于2800的，1的是CT阈值高于2800的）
    output_folder = "MA_adaptive_results"  # 输出文件夹路径
    
    create_output_dir(output_folder)
    
    # 执行MA adaptive
    print("开始MA adaptive...")
    MA_adaptive(folder_a, folder_b, folder_c, output_folder)
    print("MA adaptive完成！结果保存在:", output_folder)
