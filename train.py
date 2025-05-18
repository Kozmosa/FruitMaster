import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy

# --- 配置参数 ---
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CACHE_DIR = 'cache'
DATA_DIR = 'fruit_dataset' # 数据集根目录，请根据您的实际路径修改
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
VAL_DIR = os.path.join(DATA_DIR, 'Validation')
NUM_CLASSES = 5  # 苹果, 香蕉, 樱桃, 梨，葡萄
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 15 # 初步训练的轮数
MODEL_SAVE_PATH = 'vgg16_fruit_classifier_initial.pth' # 初步训练模型保存路径

# 如果cache目录不存在，则创建
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
torch.hub.set_dir(CACHE_DIR)

# --- 数据预处理和加载 ---
def get_dataloaders(train_dir, val_dir, batch_size):
    # VGG16期望的输入尺寸是224x224
    # ImageNet的均值和标准差
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"数据集中找到 {len(class_names)} 个类别, 但期望 {NUM_CLASSES} 个。请检查数据集文件夹。")
    print(f"检测到的类别: {class_names}")
    # 确保类别顺序与 NUM_CLASSES 对应
    # 这里假设文件夹名称就是类别名称，并且顺序是我们期望的
    # 实际应用中可能需要更复杂的映射

    return dataloaders, dataset_sizes, class_names

# --- 模型定义 ---
def get_model(num_classes):
    # 加载预训练的VGG16模型
    model_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 冻结所有卷积层的参数
    for param in model_vgg.features.parameters():
        param.requires_grad = False

    # 获取原始分类器的输入特征数
    num_ftrs = model_vgg.classifier[6].in_features
    # 替换VGG16的最后一个全连接层以匹配我们的类别数量
    model_vgg.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model_vgg

# --- 训练函数 ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25, device='cpu'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 只在训练阶段跟踪历史记录
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只在训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝模型权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"模型已保存到: {MODEL_SAVE_PATH} (验证集准确率: {best_acc:.4f})")


        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证集准确率: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# --- 主执行流程 ---
if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    print(f"使用设备: {device}")

    dataloaders, dataset_sizes, class_names_loaded = get_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)
    # 将加载的类别名称保存，以便后续脚本使用
    # 可以考虑将 class_names_loaded 保存到文件或作为全局变量传递
    print(f"训练类别: {class_names_loaded}")


    model = get_model(NUM_CLASSES)
    model = model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器 - 只优化新添加的分类层参数
    # VGG16中，model.classifier 是一个 Sequential 模块
    # 我们只希望训练 model.classifier[6] (我们新加的层)
    # 或者，更安全的方式是迭代所有参数，只将 requires_grad=True 的参数传给优化器
    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            print(f"\t将优化: {name}")

    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)

    print("开始初步训练...")
    trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=EPOCHS, device=device)
    print("初步训练完成!")
