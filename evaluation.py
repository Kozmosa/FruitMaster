import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# --- 配置参数 ---
DATA_DIR = './fruit_dataset' # 数据集根目录
# 使用验证集进行评估，或者如果您有单独的测试集，请指向测试集目录
EVAL_DIR = os.path.join(DATA_DIR, 'Test') # 或者 'test'
NUM_CLASSES = 5
BATCH_SIZE = 8
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'vgg16_fruit_classifier_initial.pth') # 或 'vgg16_fruit_classifier_initial.pth'
CLASS_NAMES = ['Apple', 'Banana', 'Cherry', 'Pear'] # 确保顺序与训练时一致

# --- 数据加载 ---
def get_eval_dataloader(eval_dir, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    eval_dataset = datasets.ImageFolder(eval_dir, eval_transform)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 获取类别名称，确保与模型训练时一致
    loaded_class_names = eval_dataset.classes
    if len(loaded_class_names) != NUM_CLASSES:
        raise ValueError(f"评估数据集中找到 {len(loaded_class_names)} 个类别, 但期望 {NUM_CLASSES} 个。")
    print(f"评估数据集中的类别: {loaded_class_names}")
    # 最好使用训练时确定的CLASS_NAMES
    return eval_dataloader, loaded_class_names # 返回从数据加载器中获取的类别名

# --- 模型加载 (与predict.py类似) ---
def load_model_for_evaluation(model_path, num_classes):
    model = models.vgg16(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- 评估函数 ---
def evaluate_model(model, dataloader, class_names_list, device='cpu'):
    model = model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- 分类报告 ---")
    # classification_report 需要目标名称
    report = classification_report(all_labels, all_preds, target_names=class_names_list, digits=4)
    print(report)

    print("\n--- 混淆矩阵 ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = ['MiSans']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list)
    plt.xlabel('预测标签 (Predicted Label)')
    plt.ylabel('真实标签 (True Label)')
    plt.title('混淆矩阵 (Confusion Matrix)')
    # 保存图像或显示
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵图像已保存为 confusion_matrix.png")
    # plt.show() # 如果在非GUI环境运行，可能需要注释掉

    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\n总体准确率 (Overall Accuracy): {accuracy:.4f}")
    return accuracy

# --- 主执行流程 ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        # 尝试从训练脚本中获取类别名称 (如果已保存)
        # with open('class_names.json', 'r') as f:
        #     CLASS_NAMES = json.load(f)
        # print(f"从文件加载的类别: {CLASS_NAMES}")
        # 如果没有，则使用上面硬编码的 CLASS_NAMES
        if len(CLASS_NAMES) != NUM_CLASSES:
            raise ValueError(f"CLASS_NAMES 列表中的类别数量 ({len(CLASS_NAMES)}) 与 NUM_CLASSES ({NUM_CLASSES}) 不匹配。")


        eval_loader, loaded_eval_class_names = get_eval_dataloader(EVAL_DIR, BATCH_SIZE)
        # 确保评估时使用的类别名称列表与模型训练时一致
        # 如果 loaded_eval_class_names 与 CLASS_NAMES 不一致或顺序不同，可能会导致指标解读错误
        # 最好的做法是在训练时保存 class_to_idx 映射，并在评估和预测时加载它。
        # 这里我们假设 CLASS_NAMES 是正确的顺序。

        model_to_evaluate = load_model_for_evaluation(MODEL_PATH, NUM_CLASSES)
        print(f"从 {MODEL_PATH} 加载模型成功。")

        evaluate_model(model_to_evaluate, eval_loader, CLASS_NAMES, device) # 使用预定义的CLASS_NAMES

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

