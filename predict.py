import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json # 用于加载类别名称

# --- 配置参数 ---
NUM_CLASSES = 5
# 类别名称应该与训练时 ImageFolder 生成的顺序一致
# 最好是从训练脚本中保存类别列表并在预测时加载
# 这里我们硬编码一个示例，实际应用中请确保这个列表是正确的
CLASS_NAMES = ['Apple', 'Banana', 'Cherry', 'Pear', 'Grape'] # 请确保顺序正确！
MODEL_PATH = 'vgg16_fruit_classifier_fine_tuned.pth' # 或 'vgg16_fruit_classifier_initial.pth'
IMAGE_TO_PREDICT = 'path/to/your/fruit_image.jpg' # 替换为你要预测的图片路径

# --- 图像预处理 ---
def preprocess_image(image_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    try:
        image = Image.open(image_path).convert('RGB') # 确保是RGB格式
        return transform(image).unsqueeze(0) #增加batch维度
    except FileNotFoundError:
        print(f"错误: 图像文件 {image_path} 未找到。")
        return None
    except Exception as e:
        print(f"加载或预处理图像时出错 {image_path}: {e}")
        return None


# --- 模型加载 ---
def load_model_for_prediction(model_path, num_classes):
    model = models.vgg16(weights=None) # 不加载预训练权重
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 根据设备加载模型权重
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval() # 设置为评估模式
    return model

# --- 预测函数 ---
def predict(model, image_tensor, class_names_list, device='cpu'):
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad(): # 不计算梯度
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0] # 获取概率
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = class_names_list[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item()
    return predicted_class, confidence

# --- 主执行流程 ---
if __name__ == '__main__':
    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"错误: 待预测的图像文件 '{IMAGE_TO_PREDICT}' 不存在。请修改 IMAGE_TO_PREDICT 变量。")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        try:
            # 尝试从训练脚本中获取类别名称 (如果已保存)
            # 例如，如果 train.py 保存了 class_names.json
            # with open('class_names.json', 'r') as f:
            #     CLASS_NAMES = json.load(f)
            # print(f"从文件加载的类别: {CLASS_NAMES}")
            # 如果没有，则使用上面硬编码的 CLASS_NAMES
            if len(CLASS_NAMES) != NUM_CLASSES:
                 raise ValueError(f"CLASS_NAMES 列表中的类别数量 ({len(CLASS_NAMES)}) 与 NUM_CLASSES ({NUM_CLASSES}) 不匹配。")

            model_to_predict = load_model_for_prediction(MODEL_PATH, NUM_CLASSES)
            print(f"从 {MODEL_PATH} 加载模型成功。")

            image_tensor = preprocess_image(IMAGE_TO_PREDICT)

            if image_tensor is not None:
                predicted_label, conf = predict(model_to_predict, image_tensor, CLASS_NAMES, device)
                print(f"预测图像: {IMAGE_TO_PREDICT}")
                print(f"预测类别: {predicted_label}")
                print(f"置信度: {conf:.4f}")

        except FileNotFoundError as e:
            print(f"错误: {e}")
        except Exception as e:
            print(f"发生错误: {e}")
