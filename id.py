import torch
from run import Net
from PIL import Image
import io
from torchvision import transforms, utils
from torch.autograd import Variable

preprocess = transforms.Compose([
   transforms.ToTensor(),
   #normalize
])
img_pil = Image.open("/home/hui/ministl_faces/ID/1.jpg")
img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
img_variable = Variable(img_tensor)
net1 = torch.load('net.pkl')
fc_out = net1(img_variable)
print fc_out
pred = torch.max(fc_out, 1)[1]
print pred
a=['Jack0','Jack1','Jack2','Jack3','Jack4','Jack5','Jack6','Jack7','Jack8','Jack9','Jack10','Jack11','Jack12','Jack13','Jack14','Jack15','Jack16','Jack17','Jack18','Jack19','Jack20','Jack21','Jack22','Jack23','Jack24','Jack25','Jack26','taohuifang','Jack28','Jack29','Jack30','Jack31','Jack32','Jack33','Jack34','Jack35','Jack35','Jack36','Jack37','Jack38','Jack39']
prednum=int(pred)
print a[prednum]
