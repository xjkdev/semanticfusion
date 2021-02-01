import torch.nn as nn
import os
import argparse


class DeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU()

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU()

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU()

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU()

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU()

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU()

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.bnfc6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Conv2d(
            in_channels=4096,
            out_channels=4096,
            kernel_size=1)
        self.bnfc7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU()

        self.fc6_deconv = nn.ConvTranspose2d(
            4096, 512, kernel_size=7, stride=1)
        self.fc6_deconv_bn = nn.BatchNorm2d(512)
        self.fc6_deconv_relu = nn.ReLU()

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.debn5_1 = nn.BatchNorm2d(512)
        self.derelu5_1 = nn.ReLU()

        self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.debn5_2 = nn.BatchNorm2d(512)
        self.derelu5_2 = nn.ReLU()

        self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.debn5_3 = nn.BatchNorm2d(512)
        self.derelu5_3 = nn.ReLU()

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv4_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.debn4_1 = nn.BatchNorm2d(512)
        self.derelu4_1 = nn.ReLU()

        self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.debn4_2 = nn.BatchNorm2d(512)
        self.derelu4_2 = nn.ReLU()

        self.deconv4_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.debn4_3 = nn.BatchNorm2d(256)
        self.derelu4_3 = nn.ReLU()

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv3_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.debn3_1 = nn.BatchNorm2d(256)
        self.derelu3_1 = nn.ReLU()

        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.debn3_2 = nn.BatchNorm2d(256)
        self.derelu3_2 = nn.ReLU()

        self.deconv3_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.debn3_3 = nn.BatchNorm2d(128)
        self.derelu3_3 = nn.ReLU()

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.debn2_1 = nn.BatchNorm2d(128)
        self.derelu2_1 = nn.ReLU()

        self.deconv2_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.debn2_2 = nn.BatchNorm2d(64)
        self.derelu2_2 = nn.ReLU()

        self.deconv1_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.debn1_1 = nn.BatchNorm2d(64)
        self.derelu1_1 = nn.ReLU()

        self.deconv1_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.debn1_2 = nn.BatchNorm2d(64)
        self.derelu1_2 = nn.ReLU()

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.class_score_nyu = nn.Conv2d(64, 14, kernel_size=1)
        self.probability = nn.Softmax2d()
        # self.argmax = nn.

    def forward(self, x):
        down1_1 = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        down1_2 = self.relu1_2(self.bn1_2(self.conv1_2(down1_1)))
        down1, mask1 = self.pool1(down1_2)

        down2_1 = self.relu2_1(self.bn2_1(self.conv2_1(down1)))
        down2_2 = self.relu2_2(self.bn2_2(self.conv2_2(down2_1)))
        down2, mask2 = self.pool2(down2_2)

        down3_1 = self.relu3_1(self.bn3_1(self.conv3_1(down2)))
        down3_2 = self.relu3_2(self.bn3_2(self.conv3_2(down3_1)))
        down3_3 = self.relu3_3(self.bn3_3(self.conv3_3(down3_2)))
        down3, mask3 = self.pool3(down3_3)

        down4_1 = self.relu4_1(self.bn4_1(self.conv4_1(down3)))
        down4_2 = self.relu4_2(self.bn4_2(self.conv4_2(down4_1)))
        down4_3 = self.relu4_3(self.bn4_3(self.conv4_3(down4_2)))
        down4, mask4 = self.pool4(down4_3)

        down5_1 = self.relu5_1(self.bn5_1(self.conv5_1(down4)))
        down5_2 = self.relu5_2(self.bn5_2(self.conv5_2(down5_1)))
        down5_3 = self.relu5_3(self.bn5_3(self.conv5_3(down5_2)))
        down5, mask5 = self.pool5(down5_3)

        down6 = self.relu6(self.bnfc6(self.fc6(down5)))

        down7 = self.relu7(self.bnfc7(self.fc7(down6)))

        up6 = self.fc6_deconv_relu(self.fc6_deconv_bn(self.fc6_deconv(down7)))

        # print(down5.shape, up6.shape, mask5.shape)
        up5 = self.unpool5(up6, mask5)
        up5_1 = self.derelu5_1(self.debn5_1(self.deconv5_1(up5)))
        up5_2 = self.derelu5_2(self.debn5_2(self.deconv5_2(up5_1)))
        up5_3 = self.derelu5_3(self.debn5_3(self.deconv5_3(up5_2)))

        # print(down4.shape, up5_3.shape, mask4.shape)
        up4 = self.unpool4(up5_3, mask4)
        up4_1 = self.derelu4_1(self.debn4_1(self.deconv4_1(up4)))
        up4_2 = self.derelu4_2(self.debn4_2(self.deconv4_2(up4_1)))
        up4_3 = self.derelu4_3(self.debn4_3(self.deconv4_3(up4_2)))

        up3 = self.unpool3(up4_3, mask3)
        up3_1 = self.derelu3_1(self.debn3_1(self.deconv3_1(up3)))
        up3_2 = self.derelu3_2(self.debn3_2(self.deconv3_2(up3_1)))
        up3_3 = self.derelu3_3(self.debn3_3(self.deconv3_3(up3_2)))

        up2 = self.unpool2(up3_3, mask2)
        up2_1 = self.derelu2_1(self.debn2_1(self.deconv2_1(up2)))
        up2_2 = self.derelu2_2(self.debn2_2(self.deconv2_2(up2_1)))

        up1 = self.unpool1(up2_2, mask1)
        up1_1 = self.derelu1_1(self.debn1_1(self.deconv1_1(up1)))
        up1_2 = self.derelu1_2(self.debn1_2(self.deconv1_2(up1_1)))

        score = self.probability(self.class_score_nyu(up1_2))

        return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict', type=str, default=os.path.join(os.path.dirname(__file__), 'state.pt'),
                        help="where model state dict is")
    args = parser.parse_args()
    net = DeConvNet()
    net.load_state_dict(torch.load(args.model))
    net.eval()
    example = torch.random([1,4, 224, 224])
    traced_model = torch.jit.trace(net, example)
    traced_model.save('model.pt')
