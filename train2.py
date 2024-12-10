#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, prefilter_voxel, render_tmp
import sys
from scene.init2 import Scene, GaussianModel
from scene.dataset_loader import GSDataset, CacheDataLoader
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


######### NOTE: customization ###########
import torch.nn as nn
import torch.nn.functional as F
from scene.PTv3.feature_predictor import FeaturePredictor
from scene.PTv3.utils.optimizers import build_optimizer, build_scheduler
from scene.PTv3.utils.metrics import psnr
from scene.PTv3.utils import loss_utils
#########################################


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, enable_tmp=True):
    enable_tmp = True
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gs_dataset = GSDataset(scene.getTrainCameraInfos(), dataset)
    gs_dataloader = CacheDataLoader(gs_dataset, max_cache_num=128, seed=42, shuffle=True, num_workers=8, batch_size=1)
    gs_testset = GSDataset(scene.getTestCameraInfos(), dataset)
    gs_testloader = CacheDataLoader(gs_testset, max_cache_num=16, seed=42, shuffle=False, num_workers=8, batch_size=1)
    ######## NOTE: build model ###########
    model = FeaturePredictor(sh_degree=dataset.sh_degree)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_path = "models/ptv3_50k.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu")[0])
        print("[INFO] Load model from {}".format(model_path))
    model = model.cuda()
    model.train()
    model_optimizer = build_optimizer(model)
    model_scheduler = build_scheduler(model_optimizer, opt.iterations)
    if enable_tmp:
        scaler = torch.cuda.amp.GradScaler()
        torch.autograd.set_detect_anomaly(False)
    else:
        torch.autograd.set_detect_anomaly(False)
    lpips_loss_func = loss_utils.lpips_loss_fn()
    ######################################
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # for iteration in range(first_iter, opt.iterations + 1):        
    iteration = first_iter
    gs_train_iter = 2000
    net_train_iter = 10000
    while iteration <= opt.iterations:
        for dataset_index, viewpoint_cam in enumerate(gs_dataloader):
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None
            
            ################ NOTE: merge attributes ################
            if iteration > 0 and iteration % (gs_train_iter + net_train_iter) == 0:
                gaussians.merge_tmp_param()
            gaussians.tmp_remove_param()
            # if iteration % (gs_train_iter + net_train_iter) <= gs_train_iter:
            #     model.eval()
            # else:
            #     model.train()
            ########################################################

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # # Pick a random Camera
            # if not viewpoint_stack:
            #     viewpoint_stack = scene.getTrainCameras().copy()
            # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            ############### NOTE: training the model ################
            if iteration % (gs_train_iter + net_train_iter) > gs_train_iter:
                visibility_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, bg)
                # gaussians_visible = gaussians.get_gs_param(visibility_mask)
                gaussians_visible = gaussians.get_all_gs_param()
                with torch.cuda.amp.autocast(enabled=enable_tmp):
                    pred_gs = model([gaussians_visible])[0]
                gaussians.tmp_update_gs_param(pred_gs)
                render_pkg = render_tmp(viewpoint_cam, gaussians, pipe, bg)
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            #########################################################

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            ################# NOTE: adapting training #################
            # Loss
            # gt_image = viewpoint_cam.original_image.cuda()
            gt_image = viewpoint_cam.original_image
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            # Ll1 = (image - gt_image).abs().mean()
            # loss = Ll1 + lpips_loss_func(image.permute(1, 2, 0).unsqueeze(0), gt_image.permute(1, 2, 0).unsqueeze(0)).mean() + psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
            # loss.backward()
            if iteration % (gs_train_iter + net_train_iter) > gs_train_iter:
                if enable_tmp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    model_optimizer.step()
                model_optimizer.zero_grad()
                model_scheduler.step()
            else:
                loss.backward()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            ###########################################################

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, gs_testloader, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, render_tmp, (pipe, background))
                # if (iteration in saving_iterations):
                #     print("\n[ITER {}] Saving Gaussians".format(iteration))
                #     scene.save(iteration)

                # # Densification
                # if iteration < opt.densify_until_iter:
                #     # Keep track of max radii in image-space for pruning
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians.reset_opacity()

                # # Optimizer step
                # if iteration < opt.iterations:
                #     gaussians.optimizer.step()
                #     gaussians.optimizer.zero_grad(set_to_none = True)

                # if (iteration in checkpoint_iterations):
                #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            iteration += 1
            if iteration > opt.iterations:
                break
            
            ### NOTE: to save the model ###
            if iteration % 10000 == 0:
                torch.save((model.state_dict(), iteration), "models/pt.pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, test_loader, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderTmpFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameraInfos()}]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # for idx, viewpoint in enumerate(config['cameras']):
                for idx, viewpoint in enumerate(test_loader):
                    image_clean = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    image = torch.clamp(renderTmpFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render_clean".format(viewpoint.image_name), image_clean[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*100 for i in range(1, 10000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
