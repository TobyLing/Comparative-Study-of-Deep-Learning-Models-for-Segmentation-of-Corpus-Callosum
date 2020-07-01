import torch

from torchvision import models

from tensorboardX import SummaryWriter

writer = SummaryWriter()

resnet = models.resnet34(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet.to(device)

dummy_input = torch.zeros(8, 3,512,512)

writer.add_graph(model, dummy_input, False)
writer.close()



