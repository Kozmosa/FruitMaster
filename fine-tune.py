import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy

# --- 配置参数 ---
DATA_DIR = 'fruits_dataset' # 数据集根目录
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
NUM_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE_FINE_TUNE = 0.00005 # 微调时使用更低的学习率
EPOCHS_FINE_TUNE = 10 # 微调的轮数
LOAD_MODEL_PATH = 'vgg16_fruit_classifier_initial.pth' # 加载初步训练的模型
FINE_TUNED_MODEL_SAVE_PATH = 'vgg16_fruit_classifier_fine_tuned.pth' # 微调后模型保存路径

# --- 数据预处理和加载 (与train.py相同) ---
def get_dataloaders(train_dir, val_dir, batch_size):
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
        raise ValueError(f"数据集中找到 {len(class_names)} 个类别, 但期望 {NUM_CLASSES} 个。")
    print(f"检测到的类别: {class_names}")
    return dataloaders, dataset_sizes, class_names

# --- 模型加载和修改 ---
def get_model_for_finetuning(num_classes, model_path):
    model_vgg = models.vgg16(weights=None) # 不加载预训练权重，因为我们将加载自己的权重

    # 修改分类器以匹配我们的类别数量
    num_ftrs = model_vgg.classifier[6].in_features
    model_vgg.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # 加载我们初步训练好的模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}。请先运行 train.py。")
    model_vgg.load_state_dict(torch.load(model_path))
    print(f"从 {model_path} 加载模型权重成功。")

    # 解冻部分卷积层进行微调
    # VGG16的 'features' 模块包含卷积层
    # 例如，我们解冻最后两个卷积块 (block4 和 block5)
    # VGG16的层结构:
    # features.0 (Conv2d) ... features.28 (Conv2d)
    # features.24 (Conv2d in block5)
    # features.17 (Conv2d in block4)
    # 让我们解冻从 block4 开始的层 (索引大约从17开始)
    # 或者更简单，解冻所有特征提取层
    # for param in model_vgg.features.parameters():
    #     param.requires_grad = True

    # 更细致地：只解冻最后几个卷积块
    # 假设我们解冻从第20个特征层开始（这只是一个例子，具体层数需要查看VGG16结构）
    # VGG16 `features` 共有30层 (0-29, 其中一些是ReLU和MaxPool)
    # 最后一个卷积块 (block5) 的卷积层是 features[24], features[26], features[28]
    # 倒数第二个卷积块 (block4) 的卷积层是 features[17], features[19], features[21]
    layers_to_unfreeze_start_index = 17 # 例如，解冻 block4 和 block5
    for i, param in enumerate(model_vgg.features.parameters()):
        if i >= layers_to_unfreeze_start_index: # 索引需要根据实际层数调整
             param.requires_grad = True
        else:
            param.requires_grad = False

    # 确保分类器层也是可训练的
    for param in model_vgg.classifier.parameters():
        param.requires_grad = True

    return model_vgg

# --- 训练函数 (与train.py类似，但用于微调) ---
def fine_tune_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, device='cpu'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 加载当前最佳准确率，如果模型之前有训练过
    # (这里我们是从初步训练模型开始，所以可以初始化为0)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), FINE_TUNED_MODEL_SAVE_PATH)
                print(f"微调模型已保存到: {FINE_TUNED_MODEL_SAVE_PATH} (验证集准确率: {best_acc:.4f})")

        print()

    time_elapsed = time.time() - since
    print(f'微调完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证集准确率 (微调后): {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- 主执行流程 ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataloaders, dataset_sizes, class_names_loaded = get_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)
    print(f"微调类别: {class_names_loaded}")

    model_ft = get_model_for_finetuning(NUM_CLASSES, LOAD_MODEL_PATH)
    model_ft = model_ft.to(device)

    # 打印哪些层是可训练的
    print("可训练的参数:")
    params_to_optimize_ft = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_optimize_ft.append(param)
            print(f"\t{name}")

    criterion = nn.CrossEntropyLoss()
    # 优化器现在应该包含解冻的卷积层和分类层的参数
    optimizer_ft = optim.SGD(params_to_optimize_ft, lr=LEARNING_RATE_FINE_TUNE, momentum=0.9)
    # 或者使用 Adam
    # optimizer_ft = optim.Adam(params_to_optimize_ft, lr=LEARNING_RATE_FINE_TUNE)


    print("开始微调模型...")
    fine_tuned_model = fine_tune_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=EPOCHS_FINE_TUNE, device=device)
    print("模型微调完成!")
