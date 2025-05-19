import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
# import json # 如果您有类别名称的json文件，可以取消注释

# --- 配置参数 ---
NUM_CLASSES = 5
# 类别名称应该与训练时 ImageFolder 生成的顺序一致
# 最好是从训练脚本中保存类别列表并在预测时加载
# 这里我们硬编码一个示例，实际应用中请确保这个列表是正确的
CLASS_NAMES = ['Apple', 'Banana', 'Cherry', 'Pear', 'Grape'] # 请确保顺序正确！
MODEL_PATH = './model/vgg16_fruit_classifier_initial.pth' # 或 'vgg16_fruit_classifier_initial.pth'
# 检查 MODEL_PATH 是否存在
if not os.path.exists(MODEL_PATH):
    print(f"警告: 模型文件 {MODEL_PATH} 未找到。请确保路径正确或文件存在。")
    # 可以选择在这里引发异常或允许程序继续（但预测会失败）
    # raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Gradio App 使用设备: {DEVICE}")

# --- 图像预处理 (修改为接收 PIL Image) ---
def preprocess_pil_image(pil_image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    try:
        # 确保是RGB格式 (Gradio的PIL输入通常已经是RGB)
        image = pil_image.convert('RGB')
        return transform(image).unsqueeze(0) #增加batch维度
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        return None

# --- 模型加载 ---
def load_model_for_prediction(model_path, num_classes):
    model = models.vgg16(weights=None) # 不加载预训练权重
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(model_path):
        # 此处错误已在全局检查，但为函数完整性保留
        print(f"错误: 模型文件 {model_path} 未找到。")
        return None

    try:
        # 根据设备加载模型权重
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # 设置为评估模式
        return model
    except Exception as e:
        print(f"加载模型权重时出错 {model_path}: {e}")
        return None

# --- 全局加载模型 ---
# 尝试从训练脚本中获取类别名称 (如果已保存)
# 例如，如果 train.py 保存了 class_names.json
# try:
#     with open('class_names.json', 'r') as f:
#         CLASS_NAMES = json.load(f)
#     print(f"从文件加载的类别: {CLASS_NAMES}")
# except FileNotFoundError:
#     print("未找到 class_names.json，使用硬编码的 CLASS_NAMES。")
# except json.JSONDecodeError:
#     print("class_names.json 文件格式错误，使用硬编码的 CLASS_NAMES。")


if len(CLASS_NAMES) != NUM_CLASSES:
    print(f"警告: CLASS_NAMES 列表中的类别数量 ({len(CLASS_NAMES)}) 与 NUM_CLASSES ({NUM_CLASSES}) 不匹配。请修正。")
    # 可以选择在这里引发异常
    # raise ValueError(f"CLASS_NAMES 列表中的类别数量 ({len(CLASS_NAMES)}) 与 NUM_CLASSES ({NUM_CLASSES}) 不匹配。")

# 尝试加载模型
loaded_model = None
if os.path.exists(MODEL_PATH):
    loaded_model = load_model_for_prediction(MODEL_PATH, NUM_CLASSES)
    if loaded_model:
        loaded_model = loaded_model.to(DEVICE)
        print(f"模型 {MODEL_PATH} 加载成功并移至 {DEVICE}。")
    else:
        print(f"无法从 {MODEL_PATH} 加载模型。预测功能将不可用。")
else:
    print(f"模型文件 {MODEL_PATH} 不存在。预测功能将不可用。")


# --- Gradio 预测接口函数 ---
def predict_interface(pil_image):
    if loaded_model is None:
        return {"错误": "模型未成功加载，无法进行预测。"}
    if pil_image is None:
        return {"错误": "请先上传一张图片。"}

    image_tensor = preprocess_pil_image(pil_image)
    if image_tensor is None:
        return {"错误": "图像预处理失败。"}

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad(): # 不计算梯度
        outputs = loaded_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0] # 获取概率

    # 创建一个包含类别名称和置信度的字典，Gradio的Label组件可以直接使用
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    return confidences

# --- 创建 Gradio 界面 ---
iface = gr.Interface(
    fn=predict_interface,
    inputs=gr.Image(type="pil", label="上传水果图片"),
    outputs=gr.Label(num_top_classes=NUM_CLASSES, label="预测结果"),
    title="水果种类分类器",
    description=(
        f"上传一张水果图片，模型将预测其种类和置信度。\n"
        f"支持的类别: {', '.join(CLASS_NAMES)}.\n"
        f"模型文件: {MODEL_PATH}"
    ),
    examples=[
        # 在这里放置一些示例图片的路径 (确保这些图片存在且可访问)
        # 例如: "sample_images/apple.jpg", "sample_images/banana.jpg"
        # 如果没有示例图片，可以注释掉 'examples' 参数或留空列表 []
    ]
)

# --- 启动 Gradio 应用 ---
if __name__ == '__main__':
    if loaded_model: # 只有模型成功加载才启动界面
        print("启动 Gradio 界面...")
        iface.launch()
    else:
        print("模型加载失败，无法启动 Gradio 界面。请检查 MODEL_PATH 和相关配置。")