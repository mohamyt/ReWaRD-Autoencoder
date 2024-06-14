import os
import random
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from args import conf
from Autoencoder import *


if __name__ == "__main__":
    # Option
    args = conf()
    print(args)

    if args.lmdb:
        from DataLoaderLMDB import Dataset_
    else:
        from DataLoader import Dataset_
    
    # Processing time
    starttime = time.time()
    today = datetime.now()
    weight_folder = "/" + today.strftime('%Y%m%d') + str(today.hour) + str(today.minute)

    # GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # to deterministic
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Training settings
    train_transform = transforms.Compose([transforms.RandomCrop((args.crop_size, args.crop_size)),
                                          transforms.ToTensor()])

    train_dataset = Dataset_(args.path2traindb, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    if args.val:
        val_transform = transforms.Compose([transforms.RandomCrop((args.crop_size, args.crop_size)),
                                            transforms.ToTensor()])
        val_dataset = Dataset_(args.path2valdb, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model & optimizer
    model = Network(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        # Load the checkpoint
        checkpoint = torch.load('checkpoint_epoch_1.pth.tar')

        # Adjust the keys of state_dict if necessary
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v

        # Load state_dict into the model
        model.load_state_dict(new_state_dict)
        args.start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        batch_losses = checkpoint.get('batch_losses', [])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    if not args.no_multigpu:
        model = nn.DataParallel(model)

    if not 'train_losses' in locals():  # Resume on existing data if available
        train_losses = []
        batch_losses = []

    # Training
    num_epochs = args.epochs
    model.to(device)

    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Save checkpoint function
    def save_checkpoint(state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs in tqdm(iterable=train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            batch_losses.append(loss.item() * imgs.size(0))

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(f"./data/weight/{args.usenet}" + weight_folder, exist_ok=True)
            checkpoint_filename = f"./data/weight/{args.usenet}{weight_folder}/checkpoint_epoch_{epoch+1}.pth.tar"
            save_checkpoint({
                'epoch': epoch + args.start_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'batch_losses': batch_losses,
            }, checkpoint_filename)
            print(f"Model checkpoint saved at epoch {epoch+1}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        # Processing time
        endtime = time.time()
        interval = endtime - starttime
        print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
        # Clear unused memory
        torch.cuda.empty_cache()

    print("Training completed.")
    # Processing time
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
