import torch
from torchvision import transforms
from PIL import Image
import sys
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_bn=True):
        x = self.conv_relu(x)
        if is_bn:
            x = self.bn(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_drop=False):
        x = self.upconv_relu(x)
        x = self.bn(x)
        if is_drop:
            x = F.dropout2d(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)

        self.up1 = Upsample(512, 512)
        self.up2 = Upsample(1024, 512)
        self.up3 = Upsample(1024, 256)
        self.up4 = Upsample(512, 128)
        self.up5 = Upsample(256, 64)

        self.last = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x6 = self.up1(x6, is_drop=True)
        x6 = torch.cat([x6, x5], dim=1)

        x6 = self.up2(x6, is_drop=True)
        x6 = torch.cat([x6, x4], dim=1)

        x6 = self.up3(x6, is_drop=True)
        x6 = torch.cat([x6, x3], dim=1)

        x6 = self.up4(x6)
        x6 = torch.cat([x6, x2], dim=1)

        x6 = self.up5(x6)
        x6 = torch.cat([x6, x1], dim=1)

        x = torch.tanh(self.last(x6))
        return x

def load_model(state_dict_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)
    
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = preprocess(image).unsqueeze(0)  
    return image

def save_output(tensor, output_path):
    unloader = transforms.ToPILImage()
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = unloader(tensor)
    tensor.save(output_path)

def main(input_image_path, output_image_path, state_dict_path):
    model = load_model(state_dict_path)
    input_image = preprocess_image(input_image_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = input_image.to(device)
    
    with torch.no_grad():
        output_image = model(input_image)
    
    save_output(output_image, output_image_path)
    print(f"Output image saved to {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test.py <input_image_path> <output_image_path> <state_dict_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        state_dict_path = sys.argv[3]
        main(input_image_path, output_image_path, state_dict_path)
