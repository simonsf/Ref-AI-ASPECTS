import numpy as np
import sys   
sys.path.append("..")
import SimpleITK as sitk 
import os, glob
import argparse
import torch
from model_seg import Seg_d  # 只需要DWI分割模型
from utils import DSC, mkdir, resize_raw_data, resize_raw_label


"""0. 环境配置"""
# 使用CPU版本
device = torch.device("cpu")

# 定义评估指标 Dice
criterion = DSC()

"""1. 初始化DWI分割模型"""
def Create_Model():
    #1. 初始化DWI分割模型
    model_d = Seg_d(in_channels=1, initial_filter_size=32, kernel_size=3, classes=1, do_instancenorm=True)

    #2. 加载DWI分割模型
    ckpts_d = r'./model/Seg_d.pkl'
    
    # 使用map_location将模型加载到CPU
    weights_d = torch.load(ckpts_d, map_location=torch.device('cpu'))
    
    model_d.load_state_dict(weights_d)
    
    # 将模型移动到CPU
    model_d = model_d.to(device)

    model = model_d  # 只返回单个模型
    print(f"Step 1: Load DWI Segmentation Model Done! Running on {device}")
    return model


"""2. 根据地址读取DWI数据, 同时进行预处理(resize+z-score)"""
def Get_Data(Path):
    
    #读取DWI数据
    data_path = Path[0] + 'dwi/' + Path[1]  # 直接指定dwi模态
    resized_data, data_dcminfo = resize_raw_data(data_path)
    slice_size = [resized_data.shape[0],resized_data.shape[1],resized_data.shape[2]]
    
    #创建数据数组
    data = np.empty(slice_size, dtype = np.float64) 
    for i in range(0,slice_size[0]):
        tmp_data = resized_data[i]  
        
        # DWI数据标准化
        tmp_data = (tmp_data - tmp_data.mean())/tmp_data.std()
        data[i] = tmp_data  
    
    #转为Tensor类型并增加维度以满足模型输入要求
    dwi_volume = torch.from_numpy(data.astype(np.float32))
    dwi_volume = torch.unsqueeze(dwi_volume, dim=1)  # 添加通道维度
    
    print("Step 2: Pre-Process DWI Data Done!")
    return dwi_volume, data_dcminfo # 返回DWI数据和成像参数


"""3. 利用DWI模型进行病灶分割"""
def Get_Segmentation(model, data):
    #设置模型为eval进行测试
    model.eval()

    with torch.no_grad():
        #DWI数据赋值并转移到CPU
        dwi_volume = data.to(device)
        
        #设置batch_size
        slices_num = data.shape[0]
        batch_size = 2
        batch_start = 0
        
        #初始化存储变量
        all_predictions = None
        
        for iter in range(0, slices_num, batch_size):
            #取出当前batch的DWI数据
            if slices_num - iter >= batch_size: 
                x_d = dwi_volume[batch_start:batch_start + batch_size]
                batch_start += batch_size
            else: 
                x_d = dwi_volume[batch_start:slices_num]
            
            #DWI数据分割
            _, pred_d = model(x_d)
            
            #对预测结果二值化
            pred_d_binary = torch.where(pred_d > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            
            #保存预测结果
            if iter == 0:
                all_predictions = pred_d_binary
            else:  
                all_predictions = torch.cat((all_predictions, pred_d_binary), dim=0)
        
        print("Step 3: DWI Segmentation Inference Done")
        return all_predictions  # 只返回分割结果


"""4. 保存分割结果"""
def Save_Result(pred_seg, data_dcminfo, output_path, patient_id):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 将tensor转换为numpy
    pred_numpy = pred_seg.cpu().numpy().squeeze()
    
    # 可选：保存为NIfTI文件
    nii_img = sitk.GetImageFromArray(pred_numpy)
    
    #保存相同的坐标空间
    nii_img.SetSpacing(data_dcminfo[0])
    nii_img.SetDirection(data_dcminfo[1])
    nii_img.SetOrigin(data_dcminfo[2])
    
    nii_file = os.path.join(output_path, f"{patient_id}_segmentation.nii.gz")
    sitk.WriteImage(nii_img, nii_file)
    
    print(f"Segmentation result saved to {nii_file}")
    return nii_file
            
            
if __name__ == '__main__':   
    
    """DWI病灶分割程序"""
    
    #定义要传入的数据路径和要分割的case名称
    base_root = r'./data/'
    P_ID = 'P2297772.nii' 
    Path = [base_root, P_ID]
    
    """Step 1, 加载DWI分割模型"""
    Model = Create_Model()
    
    """Step 2, 对传入的DWI数据进行预处理"""
    Data, data_dcminfo = Get_Data(Path)  # 只处理DWI数据
    
    """Step 3, 模型推理进行病灶分割"""
    Pred_seg = Get_Segmentation(Model, Data)  # 只进行分割
    
    """Step 4, 保存分割结果"""
    output_dir = './output'
    saved_file = Save_Result(Pred_seg, data_dcminfo, output_dir, P_ID.replace('.nii', ''))
    
    """可选：如果有标签，可以计算评估指标"""
    try:
        # 尝试加载标签用于评估
        label_path = Path[0] + 'dwi/' + P_ID.replace('.nii', 'roi.nii')
        if os.path.exists(label_path):
            Label = resize_raw_label(label_path)
            Label = torch.from_numpy(Label.astype(np.uint8))
            Label = torch.unsqueeze(Label, dim=1)
            
            # 将标签移动到CPU
            Label = Label.to(device)
            
            # 计算Dice系数
            dice_d = criterion(Pred_seg, Label)
            print(f"DWI Segmentation Dice Score: {dice_d:.4f}")
    except:
        print("No label found or error in evaluation. Only segmentation is performed.")