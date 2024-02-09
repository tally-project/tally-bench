#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
from logging import Logger
import os
import yaml
import os.path as osp
from pathlib import Path
import torch
import torch.distributed as dist
import sys
import datetime
import pathlib
import time

from utils.bench_util import wait_for_signal

curr_dir = pathlib.Path(__file__).parent.resolve()
yolo_dir = curr_dir / '../../../common/YOLOv6'
sys.path.insert(0, str(yolo_dir))

from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint, check_img_size


class TallyYoloTrainer(Trainer):

    def __init__(self, args, cfg, device, warmup_iters, total_time, total_iters, result_dict, signal, pipe):
        super().__init__(args, cfg, device)
        self.start_time = None
        self.num_iters = 0
        self.warm_iters = 0
        self.warm = False
        self.warmup_iters = warmup_iters
        self.result_dict = result_dict
        self.signal = signal
        self.total_time = total_time
        self.total_iters = total_iters
        self.finished = False
        self.time_elapsed = None
        self.pipe = pipe

    # Training Process
    def train(self):
        try:
            self.before_train_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.train_one_epoch(self.epoch)
                # self.after_epoch()

                if self.finished:
                    self.time_elapsed = self.end_time - self.start_time
            
                    if self.result_dict is not None:
                        self.result_dict["time_elapsed"] = self.time_elapsed
                        self.result_dict["iters"] = self.warm_iters

                    break

            self.strip_model()

        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    # Training loop for each epoch
    def train_one_epoch(self, epoch_num):
        try:
            for self.step, self.batch_data in self.pbar:
                self.train_in_steps(epoch_num, self.step)
                # self.print_details()
                # print(f"loss: {self.loss_items}")
                
                # Increment iterations
                self.num_iters += 1
                if self.warm:
                    self.warm_iters += 1

                    # Break if reaching total iterations
                    if self.warm_iters == self.total_iters:
                        self.finished = True
                        break

                    # Or break if time is up
                    curr_time = time.time()
                    if curr_time - self.start_time >= self.total_time:
                        self.finished = True
                        break

                if self.num_iters == self.warmup_iters:
                    self.warm = True
                    # print(f"loss: {self.loss_items}")

                    if self.signal:
                        wait_for_signal(self.pipe)

                    self.start_time = time.time()
                    print("Measurement starts ...")
        
            if self.finished:
                torch.cuda.synchronize()
                self.end_time = time.time()
                print(f"loss: {self.loss_items}")

        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise


def train_yolov6(model_name, batch_size, amp, warmup_iters, total_time,
                   total_iters=None, result_dict=None, signal=False, pipe=None):
    '''main function of training'''

    args = argparse.Namespace()
    args.data_path = str(yolo_dir / f"data/coco_new.yaml")
    args.conf_file = str(yolo_dir / f"configs/{model_name}.py")
    args.img_size = 416
    args.rect = False
    args.batch_size = batch_size
    args.epochs = 100000
    args.workers = 4
    args.device = '0'
    args.eval_interval = 20
    args.eval_final_only = False
    args.heavy_eval_range = 50
    args.check_images = False
    args.check_labels = False
    args.output_dir = './runs/train'
    args.name = 'exp'
    args.dist_url = 'env://'
    args.gpu_count = 0
    args.local_rank = -1
    args.resume = False
    args.write_trainbatch_tb = False
    args.stop_aug_last_n_epoch = 15
    args.save_ckpt_on_last_n_epoch = -1
    args.distill = False
    args.distill_feat = False
    args.quant = False
    args.calib = False
    args.teacher_model_path = None
    args.temperature = 20
    args.fuse_ab = False
    args.bs_per_gpu = 32
    args.specific_shape = False
    args.height = None
    args.width = None
    args.cache_ram = False

    # Setup
    args.local_rank, args.rank, args.world_size = get_envs()
    cfg, device, args = check_and_init(args)
    # reload envs because args was chagned in check_and_init(args)
    args.local_rank, args.rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')

    if args.local_rank != -1: # if DDP mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", \
                init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size,timeout=datetime.timedelta(seconds=7200))

    # Start
    trainer = TallyYoloTrainer(args, cfg, device, warmup_iters, total_time, total_iters, result_dict, signal, pipe)
    # PTQ
    if args.quant and args.calib:
        trainer.calibrate(cfg)
        return
    trainer.train()

    # End
    if args.world_size > 1 and args.rank == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()
    
    return trainer.time_elapsed, trainer.warm_iters

def check_and_init(args):
    '''check config files and device.'''
    # check files
    master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1
    if args.resume:
        # args.resume can be a checkpoint file path or a boolean value.
        checkpoint_path = args.resume if isinstance(args.resume, str) else find_latest_checkpoint()
        assert os.path.isfile(checkpoint_path), f'the checkpoint path is not exist: {checkpoint_path}'
        LOGGER.info(f'Resume training from the checkpoint file :{checkpoint_path}')
        resume_opt_file_path = Path(checkpoint_path).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                args = argparse.Namespace(**yaml.safe_load(f))  # load args value from args.yaml
        else:
            LOGGER.warning(f'We can not find the path of {Path(checkpoint_path).parent.parent / "args.yaml"},'\
                           f' we will save exp log to {Path(checkpoint_path).parent.parent}')
            LOGGER.warning(f'In this case, make sure to provide configuration, such as data, batch size.')
            args.save_dir = str(Path(checkpoint_path).parent.parent)
        args.resume = checkpoint_path  # set the args.resume to checkpoint path.
    else:
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.name)))
        if master_process:
            os.makedirs(args.save_dir)

    # check specific shape
    if args.specific_shape:
        if args.rect:
            LOGGER.warning('You set specific shape, and rect to True is needless. YOLOv6 will use the specific shape to train.')
        args.height = check_img_size(args.height, 32, floor=256)  # verify imgsz is gs-multiple
        args.width = check_img_size(args.width, 32, floor=256)
    else:
        args.img_size = check_img_size(args.img_size, 32, floor=256)

    cfg = Config.fromfile(args.conf_file)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')
    # check device
    device = select_device(args.device)
    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))
    # save args
    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    return cfg, device, args