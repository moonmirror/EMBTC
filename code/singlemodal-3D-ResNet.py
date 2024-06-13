import math
import os
import random
import itertools
from functools import partial
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# import matplotlib
# matplotlib.use('TkAgg')

torch.manual_seed(1020)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(1020)  # 为所有的GPU设置种子，以使得结果是确定的
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '6, 7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

global const_p
global const_p_next
global const_p_rotate


class VerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if const_p < self.p:
            img = transforms.RandomVerticalFlip(p=1)(img)
        return img


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if const_p_next < self.p:
            img = transforms.RandomHorizontalFlip(p=1)(img)
        return img


class MyRotate(object):
    def __call__(self, img):
        if const_p_rotate < 0.25:
            img = transforms.RandomRotation((-90, -90))(img)
        elif 0.25 <= const_p_rotate < 0.5:
            img = transforms.RandomRotation((-45, -45))(img)
        elif 0.5 <= const_p_rotate < 0.75:
            img = transforms.RandomRotation((45, 45))(img)
        else:
            img = transforms.RandomRotation((90, 90))(img)
        return img

# class CyclicShift(object):
#     def __init__(self, p=0.5):
#         print('use CyclicShift')
#         self.p = p
#
#     def __call__(self, img):
#         if const_p_cyclic < self.p:
#             img = self.move(img)
#         return img
#
#     def move(self, img):
#         # w, h = img.width, img.height
#         # w = min(w, h)
#         img_data = np.asarray(img)
#         img_moving = np.zeros_like(img_data)
#         if const_p_direction < 0.25:
#             img_moving[: const_p_rate] = img_data[-const_p_rate:]
#             img_moving[const_p_rate:] = img_data[: -const_p_rate]
#         elif const_p_direction < 0.5:
#             img_moving[-const_p_rate:] = img_data[: const_p_rate]
#             img_moving[: -const_p_rate] = img_data[const_p_rate:]
#         elif const_p_direction < 0.75:
#             img_moving[:, -const_p_rate:] = img_data[:, : const_p_rate]
#             img_moving[:, : -const_p_rate] = img_data[:, const_p_rate:]
#         else:
#             img_moving[:, : const_p_rate] = img_data[:, -const_p_rate:]
#             img_moving[:, const_p_rate:] = img_data[:, : -const_p_rate]
#         return Image.fromarray(img_moving)


train_compose = transforms.Compose([
    transforms.CenterCrop(160),
    VerticalFlip(),
    HorizontalFlip(),
    MyRotate(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

test_compose = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


class GetLoader(Dataset):
    def __init__(self, root, type_):
        super().__init__()
        self.root = Path(root)
        self.type = type_
        if self.type == 'train':
            self.csv = self.root / "T2_train.csv"
            self.transform = train_compose
        else:
            self.csv = self.root / "T2_test.csv"
            self.transform = test_compose

        self.csv = pd.read_csv(self.csv)
        self.csv = self.csv.dropna()
        self.csv['ID'] = self.csv['ID'].astype(str)
        self.people_classify = self.csv.loc[:, 'label']
        self.people_classify.index = self.csv['ID']
        self.people_classify = self.people_classify.to_dict()

        self.four_pic = []
        self.two_pic = []
        self.pic_files = []

        for p in self.people_classify:
            pic_file = self.root / str(p)

            if self.people_classify[p] == 4:
                self.four_pic.append(pic_file)
            else:
                self.two_pic.append(pic_file)

        if self.type == "train":
            ratio2 = int(len(self.four_pic) // len(self.two_pic))
            self.two_pic = ratio2 * self.two_pic
            self.pic_files = self.two_pic + self.four_pic
            random.shuffle(self.pic_files)
        else: 
            self.pic_files = self.four_pic + self.two_pic

        # self.pic_files.append(self.pic_files)

        # pic_file = list(pic_file.rglob('*.png'))

    def __getitem__(self, index):
        # img_single = Image.open(str(self.pic_files[index]))

        # img_t1_path = str(self.pic_files[index])
        # img_t2_path = img_t1_path.replace('T1', 'T2')
        # img_t1c_path = img_t1_path.replace('T1', 'T1c')
        # img_flair_path = img_t1_path.replace('T1', 'FLAIR')
        img_path = str(self.pic_files[index])

        img_list = os.listdir(img_path)
        ll = []
        for j in img_list:
            ll.append(int(j.split('.png')[0]))
        ll.sort()
        image = torch.zeros((155, 160, 160))

        global const_p
        const_p = np.random.random()
        global const_p_next
        const_p_next = np.random.random()
        global const_p_rotate
        const_p_rotate = np.random.random()

        for j in ll:
            imgstr = str(j) + '.png'
            img = Image.open(os.path.join(img_path, imgstr))
            img = self.transform(img)
            # print(type(img))
            # print(img.shape)
            image[j, :, :] = img
        img_data = image[13:141, :, :].unsqueeze(0)

        # print(img_data.shape)
        # img_t1 = Image.open(img_t1_path)
        # img_t2 = Image.open(img_t2_path)
        # img_t1c = Image.open(img_t1c_path)
        # img_flair = Image.open(img_flair_path)

        # image[:, :, 0] = np.array(img_t1)
        # image[:, :, 1] = np.array(img_t2)
        # image[:, :, 2] = np.array(img_t1c)
        # image[:, :, 3] = np.array(img_flair)

        # image = Image.fromarray(np.uint8(image))
        # image = img_t1

        people = str(self.pic_files[index].name)
        idd = str(people)
        y = self.people_classify[str(people)]
        if y == 2:
            one_hot = [0, 1]
        else:
            one_hot = [1, 0]
            if self.type == 'train':
                self.transform = test_compose

        one_hot = torch.tensor(one_hot)

        # new_img = np.array(img_data)
        # for j in range(91, 95):
        #     plt.subplot(2, 2, j-90)
        #     img = new_img[0, j, :, :]
        #     plt.imshow(img, cmap='gray')
        # plt.show()

        rs = {
            "img": img_data,
            "label_pre": y-2,
            "label": one_hot,
            "id": idd,
            "image_path": str(self.pic_files[index])
        }

        return rs

    def __len__(self):
        return len(self.pic_files)


# CNN模型
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AvgPool3d(
            (8, 5, 5), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


# -------------------------#
# 混淆矩阵
# -------------------------#
def confusion_matrix(preds, labels, conf_matrix, num):
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += num
    return conf_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    fig = plt.figure(figsize=None)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def statis_auc(ytrue, ypred):
    def threshold(ytrue, ypred):
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        roc_auc = metrics.auc(fpr, tpr)
        return optimal_threshold, point, fpr, tpr, roc_auc
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis = threshold(ytrue, ypred)
    return statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis


def plot_roc_curve(fpr, tpr, auc_score):
    f = plt.figure(figsize=None)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
    plt.xlabel('False Positive Rate')  # x轴标签为FPR
    plt.ylabel('True Positive Rate')  # y轴标签为TPR
    plt.title('ROC Curve')  # 设置标题
    plt.legend()
    return f
    
    
# 训练、验证、测试
def test(test_loader, path_ckpt, dict_test, dict_label):
    with torch.no_grad():
        model = resnet50()
        model = torch.nn.DataParallel(model).to(device)
        model_ckpt = torch.load(path_ckpt)
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.eval()
        model.to(device)

        loss_sum = 0
        acc_2 = 0
        acc_4 = 0
        num_class = 2
        conf_matrix = torch.zeros(num_class, num_class)

        for step, data in tqdm(enumerate(test_loader)):
            img = data['img'].to(device)
            targets = data['label'].to(device)
            idd = data['id']
            outputs = model(img).squeeze(1)
            # targets_idx = data['label_pre'].to(device)
            loss = F.cross_entropy(outputs, torch.max(targets, 1)[1]).to(device)

            loss_sum += loss.detach().item()

            # l_positive = outputs.cpu().numpy()[0][0]
            # ll_ypred.append(l_positive)

            outputs = F.softmax(outputs, 1)
            for i in range(len(idd)):
                dict_test[str(idd[i])].append(outputs[i])

            # prediction = torch.max(outputs, 1)[1]
            # pred_y = prediction.data.cpu().numpy()
            #
            # target = torch.max(targets, 1)[1]
            # target_y = target.data.cpu().numpy()
            #
            # ac = pred_y - target_y
            #
            # if target_y[0] == 1 and ac[0] == 0:
            #     acc_2 += 1
            # elif target_y[0] == 0 and ac[0] == 0:
            #     acc_4 += 1
            # conf_matrix = confusion_matrix(prediction, target, conf_matrix, 1)

        loss_avg = loss_sum / len(test_loader)

        ll_ypred = []
        ll_ytrue = []

        for key in dict_test:
            ll = torch.tensor([0.0, 0.0]).to(device)
            for i in dict_test[key]:
                ll += i

            l_positive = ll.cpu().numpy()[0]
            ll_ypred.append(l_positive)

            prediction = torch.max(ll.unsqueeze(0), 1)[1]
            pred_y = prediction.data.cpu().numpy()

            target = dict_label[key]
            if target == 2:
                target = torch.tensor([0, 1])
                ll_ytrue.append(0)
            else:
                target = torch.tensor([1, 0])
                ll_ytrue.append(1)
            target_pre = torch.max(target.unsqueeze(0), 1)[1].to(device)
            target_y = target_pre.data.cpu().numpy()
            ac = pred_y - target_y

            if target_y[0] == 1 and ac[0] == 0:
                acc_2 += 1
            elif target_y[0] == 0 and ac[0] == 0:
                acc_4 += 1
            conf_matrix = confusion_matrix(prediction, target_pre, conf_matrix, 1)

        ll_ytrue = np.array(ll_ytrue)
        ll_ypred = np.array(ll_ypred)

        optimal_threshold, point, fpr, tpr, roc_auc = statis_auc(ll_ytrue, ll_ypred)
        y_pred_binary = (ll_ypred >= optimal_threshold).astype(int)

        TP = np.sum((ll_ytrue == 1) & (y_pred_binary == 1))
        TN = np.sum((ll_ytrue == 0) & (y_pred_binary == 0))
        FP = np.sum((ll_ytrue == 1) & (y_pred_binary == 0))
        FN = np.sum((ll_ytrue == 0) & (y_pred_binary == 1))
        correct_predictions = TP + TN
        # 计算SEN（敏感性，召回率）和SPE（特异性）
        SEN = TP / (TP + FN)
        SPE = TN / (TN + FP)
        mAP = (SEN + SPE) / 2
        acc = correct_predictions / (TP + TN + FP + FN)
        
        # acc_all = (acc_2 + acc_4) / 155
        # acc_2 = acc_2 / 34
        # acc_4 = acc_4 / 121
        # acc = (acc_2 + acc_4) / 2

        return loss_avg, conf_matrix, fpr, tpr, roc_auc, SEN, SPE, mAP, acc


def train():
    epoch = 20
    # train_data = GetLoader('./Multimodal-four', 'train')
    train_data = GetLoader('./T1c_segment', 'train')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=False, pin_memory=False)
    print('train.shape: ' + str(len(train_data)))

    # test_data = GetLoader('./Multimodal-four', 'test')
    test_data = GetLoader('./T1c_segment', 'test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)
    print('test.shape: ' + str(len(test_data)))
    # for i in range(120, len(test_data)):
    #     print(test_data[i]['image_path'])

    model = resnet50()
    model = torch.nn.DataParallel(model).to(device)

    resume = False
    if resume:
        path_checkpoint = './checkpoints/best_acc.pth.tar'
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])

    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4, eps=1e-8, betas=(0.9, 0.99))

    min_acc_loss = 10
    writer = SummaryWriter(comment='Linear')

    for i_epoch in range(1, epoch+1):
        loss_sum = 0
        acc_2 = 0
        acc_4 = 0
        num_class = 2
        conf_matrix_train = torch.zeros(num_class, num_class)

        for step, data in tqdm(enumerate(train_loader)):
            img = data['img'].to(device)
            targets = data['label'].to(device)
            idd = data['id']
            # targets_idx = data['label_pre'].to(device)
            outputs = model(img).squeeze(1)
            loss = F.cross_entropy(outputs, torch.max(targets, 1)[1]).to(device)
            loss_sum += loss.detach().item()
            # outputs = F.softmax(outputs, 1)

            prediction = torch.max(outputs, 1)[1]
            pred_y = prediction.data.cpu().numpy()

            target = torch.max(targets, 1)[1]
            target_y = target.data.cpu().numpy()

            ac = pred_y - target_y

            for i in range(len(ac)):
                if target_y[i] == 1 and ac[i] == 0:
                    acc_2 += 1
                elif target_y[i] == 0 and ac[i] == 0:
                    acc_4 += 1
            conf_matrix_train = confusion_matrix(prediction, target, conf_matrix_train, 1)

            # print(pred_y)
            # print(target_y)
            # print('acc_2： ' + str(acc_2))
            # print('acc_4： ' + str(acc_4))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg = loss_sum / len(train_loader)

        fig = plot_confusion_matrix(conf_matrix_train.numpy(), classes=['HGG', 'LGG'])

        writer.add_figure('confusion matrix train', fig, global_step=i_epoch)

        acc_all = (acc_2 + acc_4) / 520
        acc_2 = acc_2 / 234
        acc_4 = acc_4 / 286
        acc = (acc_2 + acc_4) / 2

        # acc_all = (acc_2 + acc_4) / 364
        # acc_2 = acc_2 / 78
        # acc_4 = acc_4 / 286
        # acc = (acc_2 + acc_4) / 2

        train_acc = acc * 100
        train_acc_all = acc_all * 100

        print("[Epoch " + str(i_epoch) + " | " + "train loss = " + ("%.7f" % loss_avg) + ", train mAP = " + (
                    "%.3f" % train_acc) + "%, train acc = " + ("%.3f" % train_acc_all) + "%]")
        writer.add_scalar('train_loss', loss_avg, i_epoch)
        writer.add_scalar('train_mAP', train_acc, i_epoch)
        writer.add_scalar('train_acc', train_acc_all, i_epoch)

        # 保存
        path_ckpt = r"./checkpoints/" + str(i_epoch) + ".pth.tar"
        torch.save({"epoch": i_epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, path_ckpt)

        csv_file_2 = pd.read_csv("./test.csv")
        dict_test = csv_file_2.loc[:, 'label'].map(lambda x: [])
        dict_test.index = csv_file_2['ID']
        dict_test = dict_test.to_dict()

        dict_label_test = csv_file_2.loc[:, 'label']
        dict_label_test.index = csv_file_2['ID']
        dict_label_test = dict_label_test.to_dict()

        loss_test, confmatrix, fpr, tpr, roc_auc, SEN, SPE, acc_sum, acc_all = test(test_loader, path_ckpt, dict_test, dict_label_test)
        test_acc = acc_sum * 100
        test_acc_all = acc_all * 100

        print("[Epoch " + str(i_epoch) + " | " + "test loss = " + ("%.7f" % loss_test) + ", test mAP = " + ("%.2f" % test_acc) + \
              "%, test acc = " + ("%.2f" % test_acc_all) + "%, test auc = " + ("%.4f" % roc_auc) + ", test SEN = " + ("%.4f" % SEN) \
              + ", test SPE = " + ("%.4f" % SPE) + "]")

        writer.add_scalar('test_loss', loss_test, i_epoch)
        writer.add_scalar('test_mAP', test_acc, i_epoch)
        writer.add_scalar('test_acc', test_acc_all, i_epoch)
        writer.add_scalar('test_auc', roc_auc, i_epoch)

        fig = plot_confusion_matrix(confmatrix.numpy(), classes=['HGG', 'LGG'])

        writer.add_figure('confusion matrix test', fig, global_step=i_epoch)

        f = plot_roc_curve(fpr, tpr, roc_auc)
        writer.add_figure('roc test', f, global_step=i_epoch)
        fpr_str = ' '.join(map(str, fpr.ravel().tolist()))
        tpr_str = ' '.join(map(str, tpr.ravel().tolist()))
        writer.add_text('fpr test', fpr_str, global_step=i_epoch)
        writer.add_text('tpr test', tpr_str, global_step=i_epoch)

        if loss_test < min_acc_loss:
            path_ckpt_best = r"./checkpoints/best_acc.pth.tar"
            torch.save({"epoch": i_epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, path_ckpt_best)
            min_acc_loss = loss_test
            print("最优epoch更新为：" + str(i_epoch))

        writer.close()


if __name__ == '__main__':
    train()
