import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
import matplotlib.pyplot as plt # 新增：导入matplotlib

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

# 图表保存路径
ACC_PLOT_PATH = 'Figure_4_1_1_Preliminary_Training_Accuracy_Curve.png'
LOSS_PLOT_PATH = 'Figure_4_1_2_Preliminary_Training_Loss_Curve.png'


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
    return dataloaders, dataset_sizes, class_names

# --- 模型定义 ---
def get_model(num_classes):
    model_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model_vgg.features.parameters():
        param.requires_grad = False
    num_ftrs = model_vgg.classifier[6].in_features
    model_vgg.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model_vgg

# --- 训练函数 (已修改以包含绘图) ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25, device='cpu'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 用于存储历史数据
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 初始化绘图窗口
    plt.ion() # 开启交互模式
    fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
    fig_loss, ax_loss = plt.subplots(figsize=(8, 6))

    # 设置准确率图的初始标题和标签 (中文 + English)
    ax_acc.set_title('准确率曲线 / Accuracy Curve')
    ax_acc.set_xlabel('轮次 / Epoch')
    ax_acc.set_ylabel('准确率 / Accuracy')
    ax_acc.grid(True)

    # 设置损失图的初始标题和标签
    ax_loss.set_title('损失曲线 / Loss Curve')
    ax_loss.set_xlabel('轮次 / Epoch')
    ax_loss.set_ylabel('损失 / Loss')
    ax_loss.grid(True)
    
    # 预先绘制图例，避免在循环中重复创建图例对象导致警告
    line_train_acc, = ax_acc.plot([], [], 'r-o', label='训练集准确率 (Train Acc)')
    line_val_acc, = ax_acc.plot([], [], 'b-o', label='验证集准确率 (Validation Acc)')
    ax_acc.legend()

    line_train_loss, = ax_loss.plot([], [], 'r-o', label='训练集损失 (Train Loss)')
    line_val_loss, = ax_loss.plot([], [], 'b-o', label='验证集损失 (Validation Loss)')
    ax_loss.legend()


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_metrics = {} # 存储当前epoch的 train/val loss/acc

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
            
            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc.item() # .item()转为python float

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"模型已保存到: {MODEL_SAVE_PATH} (验证集准确率: {best_acc:.4f})")

        # 记录当前epoch的数据
        history['train_loss'].append(epoch_metrics['train_loss'])
        history['train_acc'].append(epoch_metrics['train_acc'])
        history['val_loss'].append(epoch_metrics['val_loss'])
        history['val_acc'].append(epoch_metrics['val_acc'])

        # 更新绘图
        epochs_range = range(1, epoch + 2)

        # 更新准确率图数据
        line_train_acc.set_data(epochs_range, history['train_acc'])
        line_val_acc.set_data(epochs_range, history['val_acc'])
        ax_acc.relim() # 重新计算数据范围
        ax_acc.autoscale_view(True,True,True) # 自动调整坐标轴
        ax_acc.set_title(f'准确率曲线 (Epoch {epoch+1}/{num_epochs})')
        fig_acc.canvas.draw_idle() # 更新图形显示

        # 更新损失图数据
        line_train_loss.set_data(epochs_range, history['train_loss'])
        line_val_loss.set_data(epochs_range, history['val_loss'])
        ax_loss.relim()
        ax_loss.autoscale_view(True,True,True)
        ax_loss.set_title(f'损失曲线 (Epoch {epoch+1}/{num_epochs})')
        fig_loss.canvas.draw_idle()

        plt.pause(0.1) # 暂停一小段时间以允许图形更新

        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证集准确率: {best_acc:4f}')

    # 保存最终的图表
    ax_acc.set_title('初步训练阶段准确率曲线（训练集与验证集）') # 最终标题
    fig_acc.savefig(ACC_PLOT_PATH)
    print(f"准确率曲线图已保存到: {ACC_PLOT_PATH}")

    ax_loss.set_title('初步训练阶段损失曲线（训练集与验证集）') # 最终标题
    fig_loss.savefig(LOSS_PLOT_PATH)
    print(f"损失曲线图已保存到: {LOSS_PLOT_PATH}")

    plt.ioff() # 关闭交互模式

    model.load_state_dict(best_model_wts)
    return model, history


# --- 主执行流程 ---
if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # 为了确保在没有GPU的环境下也能运行和绘图，这里强制使用CPU
                                  # 如果您有CUDA环境并且matplotlib后端支持，可以改回自动检测
    
    print(f"使用设备: {device}")

    dataloaders, dataset_sizes, class_names_loaded = get_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)
    print(f"训练类别: {class_names_loaded}")

    model = get_model(NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            print(f"\t将优化: {name}")

    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)

    print("开始初步训练...")
    # train_model 现在返回模型和历史记录
    trained_model, training_history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=EPOCHS, device=device)
    print("初步训练完成!")

    # print("\n训练历史:")
    # for key in training_history:
    #     print(f"{key}: {training_history[key]}")

    plt.show() # 显示最终的图表，并保持窗口直到手动关闭