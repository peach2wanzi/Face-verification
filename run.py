import torch
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

root="/home/hui/ministl_faces/"
LR=0.01

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
train_data = MyDataset(txt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=20)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =torch. nn.Sequential(  # input shape (3, 92, 112)wide and hight
            torch.nn.Conv2d(    #filter
                in_channels=3, 
                out_channels=32,  # the number of filter
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # padding=(kernel_size-1)/2  when stride=1,tian'chong'ceng
            ),  # output shape (32, 92, 112)
            torch.nn.ReLU(),  
            torch.nn.MaxPool2d(kernel_size=2), #output shape (32, 46, 56)
        )
        self.conv2 =torch. nn.Sequential(  # input shape (32, 46, 56)
            torch.nn.Conv2d(32, 64, 5, 1, 2),  # output shape (64, 46, 56)
            torch.nn.ReLU(),  # activation
            torch.nn.MaxPool2d(2),  # output shape (64, 23, 28)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 *23*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 40)
        )

    def forward(self, x): 
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1) 
        out = self.dense(res)
        return out

model = Net()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(40):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))
 # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
torch.save(model, 'net.pkl')
