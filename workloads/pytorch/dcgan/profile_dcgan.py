
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.utils.data.distributed

import os
from torch.nn import DataParallel
from torchvision import transforms

from utils.bench_util import wait_for_signal

# Training
def benchmark_dcgan(model_name, batch_size, amp, warmup_iters, total_time,
                    total_iters=None, result_dict=None, signal=False,
                    imageSize=64, nz=100, ngf=64, ndf=64, arg_netG='',
                    arg_netD='', lr=0.0002, beta1=0.5):
    device = 'cuda'

    dataset = dset.FakeData(size=30000, image_size=(3, imageSize, imageSize),
                            transform=transforms.ToTensor())
    nc = 3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ngpu = 1
    nz = int(nz)
    ngf = int(ngf)
    ndf = int(ndf)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output


    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if arg_netG != '':
        netG.load_state_dict(torch.load(arg_netG.netG))

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if arg_netD != '':
        netD.load_state_dict(torch.load(arg_netD))

    if amp:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    compile_options = {
        "epilogue_fusion": True,
        "max_autotune": True,
        "triton.cudagraphs": False,
    }

    netG = torch.compile(netG, backend='inductor', options=compile_options)
    netD = torch.compile(netD, backend='inductor', options=compile_options)

    start_time = None
    num_iters = 0
    warm_iters = 0
    warm = False

    while True:
        stop = False

        for i, data in enumerate(dataloader, 0):
            if amp:
                with torch.cuda.amp.autocast():
                    # train with real
                    netD.zero_grad()
                    real_cpu = data[0].to(device)
                    batch_size = real_cpu.size(0)
                    label = torch.full((batch_size,), real_label,
                                    dtype=real_cpu.dtype, device=device)
                    output = netD(real_cpu)
                    errD_real = criterion(output, label)
                scaler.scale(errD_real).backward()

                with torch.cuda.amp.autocast():
                    # train with fake
                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake.detach())
                    errD_fake = criterion(output, label)
                    errD = errD_real + errD_fake
                scaler.scale(errD_fake).backward()
                scaler.step(optimizerD)

                with torch.cuda.amp.autocast():
                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    output = netD(fake)
                    errG = criterion(output, label)   
                scaler.scale(errG).backward()    
                scaler.step(optimizerG)
                scaler.update()
            else:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label,
                                dtype=real_cpu.dtype, device=device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()

                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()

            # Increment iterations
            num_iters += 1
            if warm:
                warm_iters += 1

                # Break if reaching total iterations
                if warm_iters == total_iters:
                    stop = True
                    break

                # Or break if time is up
                curr_time = time.time()
                if curr_time - start_time >= total_time:
                    stop = True
                    break

            if num_iters == warmup_iters:
                warm = True

                if signal:
                    wait_for_signal()

                start_time = time.time()
                print("Measurement starts ...")
        
        if stop:
            break

    end_time = time.time()
    time_elapsed = end_time - start_time
    
    if result_dict is not None:
        result_dict["time_elapsed"] = time_elapsed
        result_dict["iters"] = warm_iters

    return time_elapsed, warm_iters