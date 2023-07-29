from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from evaluation import *
from unetmode import *
from PIL import Image
import torch

class Mydataset(Dataset):  # 获取文件夹里的图片，并返回图片和label

    def __init__(self, root_dir, label_dir, transform=None, ltransform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir  #
        self.imagepath = os.path.join(self.root_dir)
        self.labpath = os.path.join(self.label_dir)
        self.imgs_path = os.listdir(self.imagepath)
        self.lab_path = os.listdir(self.labpath)
        self.transform = transform
        self.ltransform = ltransform

    def __getitem__(self, idx):
        img_name = self.imgs_path[idx]
        img_image_path = os.path.join(self.root_dir, img_name)
        lab_name = self.lab_path[idx]
        lab_label_path = os.path.join(self.label_dir, lab_name)

        img = Image.open(img_image_path).convert('RGB')
        lab = Image.open(lab_label_path).convert('1')

        if self.transform is not None:
            img = self.transform(img)
        if self.ltransform is not None:
            lab = self.ltransform(lab)
        return img, lab

    def __len__(self):
        return len(self.imgs_path)


trainimage = "D:/cell/images"
trainlab = "D:/cell/masks/0"

imagetransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
labtransform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()
     ]
)
data = Mydataset(trainimage, trainlab, transform=imagetransform, ltransform=labtransform)
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size], torch.manual_seed(100))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=4)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = U_Net(3,1).to(device)

num_epochs = 1
learning_rate = 0.0001  # 学习率
# 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=False,
                                                       threshold=0.0001, min_lr=0)  # 固定步长衰减
criterion = torch.nn.CrossEntropyLoss()
best = 0.
besti = 0
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 50)

    # 执行学习率指数衰减
    length = 0
    epoch_loss = 0
    step = 0
    IoU = 0
    DC = 0
    for i, (images, GT) in enumerate(train_loader):
        images = images.to(device)
        GT = GT.to(device)
        optimizer.zero_grad()  # 梯度清理
        SR = model(images)
        loss = criterion(SR, GT)
        loss.backward(loss)  # 误差反向传播
        optimizer.step()  # 基于当前更新梯度
        epoch_loss += loss.item()
        IoU += iou_score(SR, GT)  # jaccard 相似性
        DC += dice_coef(SR, GT)  # Dice
        length += images.size(0)
    scheduler.step(loss)
    IoU = IoU / length
    DC = DC / length
    # IOU=IOU / length
    print(
        'Epoch [%d/%d], Loss: %.4f, [Training] IoU: %.4f, DC: %.4f' % (
            epoch + 1, num_epochs, \
            epoch_loss, \
             IoU,DC))
    with torch.no_grad():
        model.eval()

        IoU = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        length = 0
        for i, (images, GT,) in enumerate(test_loader):
            images = images.to(device)
            GT = GT.to(device)
            SR = model(images)
            IoU += iou_score(SR, GT)
            DC += dice_coef(SR, GT)
            length += images.size(0)
        IoU = IoU / length
        DC = DC / length
        if DC > best:
            best = DC
            besti = epoch + 1
            torch.save({'epoch':besti,
         'model_state_dict':model.state_dict(),
         'otpimizer_state_dict':optimizer.state_dict()
         },'filename'
                       )

        print('[test]  IoU: %.4f, DC: %.4f' % (
             IoU, DC))

print('[bestepoch and bestdice] epoch:%.4d bestDC: %.4f' % (
    besti, best))
