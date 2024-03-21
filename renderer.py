import numpy as np
import random
import os, imageio
from tqdm.auto import tqdm
from utils import *
from models.relight_utils import render_with_BRDF
import torch
import torchvision.utils as vutils


@torch.no_grad()
def compute_rescale_ratio(tensoIR, dataset, sampled_num=20):
    '''compute three channel rescale ratio for albedo by sampling some views
    - Args:
        tensoIR: model
        dataset: dataset containing the G.T albedo
    - Returns:
        single_channel_ratio: median of the ratio of the first channel
        three_channel_ratio: median of the ratio of the three channels
    '''
    W, H = dataset.img_wh
    data_num = len(dataset)
    interval = data_num // sampled_num
    idx_list = [i * interval for i in range(sampled_num)]
    ratio_list = list()
    gt_albedo_list = []
    reconstructed_albedo_list = []
    for idx in tqdm(idx_list, desc="compute rescale ratio"):
        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(tensoIR.device) # [H*W, 6]
        gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu() # [H*W]
        gt_albedo = item['albedo'].squeeze(0).to(tensoIR.device) # [H*W, 3]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(tensoIR.device).fill_(0)
        albedo_map = list()
        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), 3000) 
        for chunk_idx in chunk_idxs:
            rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                fresnel_chunk, acc_chunk, *temp \
                = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)
            albedo_map.append(albedo_chunk.detach())
        albedo_map = torch.cat(albedo_map, dim=0).reshape(H, W, 3)
        gt_albedo = gt_albedo.reshape(H, W, 3)
        gt_mask = gt_mask.reshape(H, W)
        gt_albedo_list.append(gt_albedo[gt_mask])
        reconstructed_albedo_list.append(albedo_map[gt_mask])
    # ratio = torch.stack(ratio_list, dim=0).mean(dim=0)
    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    single_channel_ratio = (gt_albedo_all / albedo_map_all.clamp(min=1e-6))[..., 0].median()
    three_channel_ratio, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)
    print("single channel rescale ratio: ", single_channel_ratio)
    print("three channels rescale ratio: ", three_channel_ratio)
    return single_channel_ratio, three_channel_ratio



def Renderer_TensoIR_train(  
                            rays=None, 
                            normal_gt=None, 
                            light_idx=None, 
                            tensoIR=None, 
                            N_samples=-1,
                            ndc_ray=False, 
                            white_bg=True, 
                            is_train=False,
                            is_relight=True,
                            sample_method='fixed_envirmap',
                            chunk_size=15000,
                            device='cuda',      
                            args=None,
                        ):

   
    rays = rays.to(device)
    light_idx = light_idx.to(device, torch.int32)
    rgb_map, depth_map, normal_map, albedo_map, roughness_map, \
        fresnel_map, acc_map, normals_diff_map, normals_orientation_loss_map, \
        acc_mask, albedo_smoothness_loss, roughness_smoothness_loss \
        = tensoIR(rays, light_idx, is_train=is_train, white_bg=white_bg, is_relight=is_relight, ndc_ray=ndc_ray, N_samples=N_samples)

    # If use GT normals
    if tensoIR.normals_kind == "gt_normals" and normal_gt is not None:
        normal_map = normal_gt.to(device)

    # Physically-based Rendering(Relighting)
    if is_relight:
        rgb_with_brdf_masked = render_with_BRDF(   
                                                depth_map[acc_mask],
                                                normal_map[acc_mask],
                                                albedo_map[acc_mask],
                                                roughness_map[acc_mask].repeat(1, 3),
                                                fresnel_map[acc_mask],
                                                rays[acc_mask],
                                                tensoIR,
                                                light_idx[acc_mask],
                                                sample_method,
                                                chunk_size=chunk_size,
                                                device=device,
                                                args=args
                                               )




        rgb_with_brdf = torch.ones_like(rgb_map) # background default to be white
        rgb_with_brdf[acc_mask] = rgb_with_brdf_masked
        # rgb_with_brdf = rgb_with_brdf * acc_map[..., None]  + (1. - acc_map[..., None])
    else:
        rgb_with_brdf = torch.ones_like(rgb_map)


    ret_kw = {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "normal_map": normal_map,
        "albedo_map": albedo_map,
        "acc_map": acc_map,
        "roughness_map": roughness_map,
        "fresnel_map": fresnel_map,
        'rgb_with_brdf_map': rgb_with_brdf,
        'normals_diff_map': normals_diff_map,
        'normals_orientation_loss_map': normals_orientation_loss_map,
        'albedo_smoothness_loss': albedo_smoothness_loss,
        'roughness_smoothness_loss': roughness_smoothness_loss,
    }

    return ret_kw






@torch.no_grad()
def evaluation_iter_TensoIR(
        test_dataset,
        tensoIR,
        args,
        renderer,
        savePath=None,
        prtx='',
        N_samples=-1,
        white_bg=False,
        ndc_ray=False,
        compute_extra_metrics=True,
        device='cuda',
        logger=None,
        step=None,
        test_all=False,
):

    
    PSNRs_rgb, rgb_maps, depth_maps, gt_maps, gt_rgb_brdf_maps = [], [], [], [], []
    PSNRs_rgb_brdf = []
    rgb_with_brdf_maps, normal_rgb_maps, normal_rgb_vis_maps, normals_rgb_gt_maps = [], [], [], []
    albedo_maps, single_aligned_albedo_maps, three_aligned_albedo_maps, gt_albedo_maps, roughness_maps, fresnel_maps, normals_diff_maps, normals_orientation_loss_maps  = [], [], [], [], [], [], [], []
    normal_raw_list = []
    normal_gt_list = []
    ssims, l_alex, l_vgg = [], [], []
    ssims_rgb_brdf, l_alex_rgb_brdf, l_vgg_rgb_brdf = [], [], []
    ssims_albedo_single, l_alex_albedo_single, l_vgg_albedo_single = [], [], []
    ssims_albedo_three, l_alex_albedo_three, l_vgg_albedo_three = [], [], []

    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/nvs_with_radiance_field", exist_ok=True)
    os.makedirs(savePath + "/nvs_with_brdf", exist_ok=True)
    os.makedirs(savePath + "/normal", exist_ok=True)
    os.makedirs(savePath + "/normal_vis", exist_ok=True)
    os.makedirs(savePath + "/brdf", exist_ok=True)
    os.makedirs(savePath + "/envir_map/", exist_ok=True)
    os.makedirs(savePath + "/acc_map", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh

    num_test = len(test_dataset) if test_all else min(args.N_vis, len(test_dataset))

    gt_envir_map = None
    if test_dataset.lights_probes is not None:
        gt_envir_map = test_dataset.lights_probes.reshape(test_dataset.envir_map_h, test_dataset.envir_map_w, 3).numpy()
        gt_envir_map = np.uint8(np.clip(np.power(gt_envir_map, 1./2.2), 0., 1.) * 255.)
        
        # resize to 256 * 512
        gt_envir_map = cv2.resize(gt_envir_map, (512, 256), interpolation=cv2.INTER_CUBIC)

    _, view_dirs = tensoIR.generate_envir_map_dir(256, 512)

    predicted_envir_map = tensoIR.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
    predicted_envir_map = predicted_envir_map.reshape(256, 512, 3).cpu().detach().numpy()
    predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
    predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
    if gt_envir_map is not None:
        envirmap = np.concatenate((gt_envir_map, predicted_envir_map), axis=1)
    else:
        envirmap = predicted_envir_map
    # save predicted envir map
    imageio.imwrite(f'{savePath}/envir_map/{prtx}envirmap.png', envirmap)
    test_duration = int(len(test_dataset) / num_test)

    # compute global rescale ratio for predicted albedo
    if test_all:
        global_rescale_value_single, global_rescale_value_three = compute_rescale_ratio(tensoIR, test_dataset, sampled_num=20)
        global_rescale_value_single, global_rescale_value_three = global_rescale_value_single.cpu(), global_rescale_value_three.cpu()


    for idx in range(num_test):
        if test_all:
            print(f"test {idx} / {num_test}")
        item = test_dataset.__getitem__(idx * test_duration)
        rays = item['rays']                 # [H*W, 6]
        gt_rgb = item['rgbs'][0]            # [H*W, 3]
        light_idx = item['light_idx'][0]    # [H*W, 1]
        gt_normals = item['normals']        # [H*W, 3]
        gt_rgb_wirh_brdf = gt_rgb           # [H*W, 3]
        gt_mask = item['rgbs_mask']         # [H*W, 1]
        albedo_gt = item['albedo']          # [H*W, 3]
        rgb_map, acc_map, depth_map, normal_map, albedo_map, roughness_map = [], [], [], [], [], []
        fresnel_map, rgb_with_brdf_map, normals_diff_map, normals_orientation_loss_map = [], [], [], []
        
        chunk_idxs = torch.split(torch.arange(rays.shape[0]), args.batch_size_test)
        for chunk_idx in chunk_idxs:
            ret_kw= renderer(   
                                rays[chunk_idx], 
                                None, # not used
                                light_idx[chunk_idx],
                                tensoIR, 
                                N_samples=N_samples,
                                ndc_ray=ndc_ray,
                                white_bg=white_bg,
                                sample_method='fixed_envirmap',
                                chunk_size=args.relight_chunk_size,  
                                device=device,
                                args=args
                            )
            rgb_map.append(ret_kw['rgb_map'].detach().cpu())
            depth_map.append(ret_kw['depth_map'].detach().cpu())
            normal_map.append(ret_kw['normal_map'].detach().cpu())
            albedo_map.append(ret_kw['albedo_map'].detach().cpu())
            roughness_map.append(ret_kw['roughness_map'].detach().cpu())
            fresnel_map.append(ret_kw['fresnel_map'].detach().cpu())
            rgb_with_brdf_map.append(ret_kw['rgb_with_brdf_map'].detach().cpu())
            normals_diff_map.append(ret_kw['normals_diff_map'].detach().cpu())
            normals_orientation_loss_map.append(ret_kw['normals_orientation_loss_map'].detach().cpu())
            acc_map.append(ret_kw['acc_map'].detach().cpu())

        
        rgb_map = torch.cat(rgb_map)
        depth_map = torch.cat(depth_map)
        normal_map = torch.cat(normal_map)
        albedo_map = torch.cat(albedo_map)
        roughness_map = torch.cat(roughness_map)
        fresnel_map = torch.cat(fresnel_map)
        rgb_with_brdf_map = torch.cat(rgb_with_brdf_map)
        normals_diff_map = torch.cat(normals_diff_map)
        normals_orientation_loss_map = torch.cat(normals_orientation_loss_map)
        acc_map = torch.cat(acc_map)

        # normal_map_to_test = acc_map[..., None] * normal_map + (1 - acc_map[..., None]) * torch.tensor([0.0, 0.0, 1.0])

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_with_brdf_map = rgb_with_brdf_map.clamp(0.0, 1.0)

        acc_map = acc_map.reshape(H, W).detach().cpu()

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).detach().cpu(), depth_map.reshape(H, W).detach().cpu()
        rgb_with_brdf_map = rgb_with_brdf_map.reshape(H, W, 3).detach().cpu()
        albedo_map = albedo_map.reshape(H, W, 3).detach().cpu()

        single_aligned_albedo_map = torch.ones_like(albedo_map)
        three_aligned_albedo_map = torch.ones_like(albedo_map)
        gt_albedo_reshaped = albedo_gt.reshape(H, W, 3).detach().cpu()
        gt_mask_reshaped = gt_mask.reshape(H, W).detach().cpu()
        # single channel alignment for albedo
        if test_all:
            ratio_value = global_rescale_value_single
        else:
            ratio_value = (gt_albedo_reshaped[gt_mask_reshaped] / albedo_map[gt_mask_reshaped].clamp(min=1e-6))[..., 0].median()
        single_aligned_albedo_map[gt_mask_reshaped] = (ratio_value * albedo_map[gt_mask_reshaped]).clamp(min=0.0, max=1.0)
        # three channel alignment for albedo
        if test_all:
            ratio_value = global_rescale_value_three
        else:
            ratio_value, _ = (gt_albedo_reshaped[gt_mask_reshaped]/ albedo_map[gt_mask_reshaped].clamp(min=1e-6)).median(dim=0)
        three_aligned_albedo_map[gt_mask_reshaped] = (ratio_value * albedo_map[gt_mask_reshaped]).clamp(min=0.0, max=1.0)


        roughness_map = roughness_map.reshape(H, W, 1).repeat(1, 1, 3).detach().cpu()
        fresnel_map = fresnel_map.reshape(H, W, 3).detach().cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        # Store loss and images
        if test_dataset.__len__():
            gt_rgb = gt_rgb.view(H, W, 3)
            gt_rgb_wirh_brdf = gt_rgb_wirh_brdf.view(H, W, 3)
            loss_rgb = torch.mean((rgb_map - gt_rgb) ** 2)
            loss_rgb_brdf = torch.mean((rgb_with_brdf_map - gt_rgb_wirh_brdf) ** 2)
            PSNRs_rgb.append(-10.0 * np.log(loss_rgb.item()) / np.log(10.0))
            PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensoIR.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensoIR.device)

                ssim_rgb_brdf = rgb_ssim(rgb_with_brdf_map, gt_rgb_wirh_brdf, 1)
                l_a_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'alex', tensoIR.device)
                l_v_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'vgg', tensoIR.device)

                # single channel aligned albedo
                ssim_albedo_single = rgb_ssim(single_aligned_albedo_map, gt_albedo_reshaped, 1)
                l_a_albedo_single = rgb_lpips(gt_albedo_reshaped.numpy(), single_aligned_albedo_map.numpy(), 'alex', tensoIR.device)
                l_v_albedo_single = rgb_lpips(gt_albedo_reshaped.numpy(), single_aligned_albedo_map.numpy(), 'vgg', tensoIR.device)
                # three channel aligned albedo
                ssim_albedo_three = rgb_ssim(three_aligned_albedo_map, gt_albedo_reshaped, 1)
                l_a_albedo_three = rgb_lpips(gt_albedo_reshaped.numpy(), three_aligned_albedo_map.numpy(), 'alex', tensoIR.device)
                l_v_albedo_three = rgb_lpips(gt_albedo_reshaped.numpy(), three_aligned_albedo_map.numpy(), 'vgg', tensoIR.device)

                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

                ssims_rgb_brdf.append(ssim_rgb_brdf)
                l_alex_rgb_brdf.append(l_a_rgb_brdf)
                l_vgg_rgb_brdf.append(l_v_rgb_brdf)

                ssims_albedo_single.append(ssim_albedo_single)
                l_alex_albedo_single.append(l_a_albedo_single)
                l_vgg_albedo_single.append(l_v_albedo_single)

                ssims_albedo_three.append(ssim_albedo_three)
                l_alex_albedo_three.append(l_a_albedo_three)
                l_vgg_albedo_three.append(l_v_albedo_three)




        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_with_brdf_map = (rgb_with_brdf_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        gt_rgb_wirh_brdf = (gt_rgb_wirh_brdf.numpy() * 255).astype('uint8')
        albedo_map = (albedo_map.numpy() * 255).astype('uint8')
        roughness_map = (roughness_map.numpy() * 255).astype('uint8')
        fresnel_map = (fresnel_map.numpy() * 255).astype('uint8')
        acc_map = (acc_map.numpy() * 255).astype('uint8')

        # Visualize normal
        ## Prediction
        normal_map = F.normalize(normal_map, dim=-1)
        # normal_map_to_test = F.normalize(normal_map_to_test, dim=-1)
        # normal_raw_list.append(normal_map_to_test)
        # normal_rgb_map = normal_map_to_test * 0.5 + 0.5


        normal_raw_list.append(normal_map)

        normal_rgb_map = normal_map * 0.5 + 0.5 # map from [-1, 1] to [0, 1] to visualize
        normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype('uint8')
        normal_rgb_vis_map = (normal_rgb_map * (acc_map[:, :, None] / 255.0) + (1 -(acc_map[:, :, None] / 255.0)) * 255).astype('uint8') # white background


        # GT normal 
        gt_normals = F.normalize(gt_normals, dim=-1)
        normal_gt_list.append(gt_normals)
        gt_normals_rgb_map = gt_normals * 0.5 + 0.5
        gt_normals_rgb_map = (gt_normals_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')

        # difference between the predicted normals and derived normals
        normals_diff_map = (torch.clamp(normals_diff_map, 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        # normals orientation loss map
        normals_orientation_loss_map = (torch.clamp(normals_orientation_loss_map , 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        rgb_with_brdf_maps.append(rgb_with_brdf_map)
        depth_maps.append(depth_map)
        gt_maps.append(gt_rgb)
        gt_rgb_brdf_maps.append(gt_rgb_wirh_brdf)
        normal_rgb_maps.append(normal_rgb_map)
        normal_rgb_vis_maps.append(normal_rgb_vis_map)
        normals_rgb_gt_maps.append(gt_normals_rgb_map)
        if not test_all:
            normals_diff_maps.append(normals_diff_map)
            normals_orientation_loss_maps.append(normals_orientation_loss_map)


        albedo_maps.append(albedo_map)
        single_aligned_albedo_maps.append((single_aligned_albedo_map.numpy())**(1/2.2))
        three_aligned_albedo_maps.append((three_aligned_albedo_map.numpy())**(1/2.2))
        gt_albedo_maps.append((gt_albedo_reshaped.numpy())**(1/2.2))
        roughness_maps.append(roughness_map)
        fresnel_maps.append(fresnel_map)


        if savePath is not None:
            rgb_map = np.concatenate((rgb_map, gt_rgb, depth_map), axis=1)
            rgb_with_brdf_map = np.concatenate((rgb_with_brdf_map, gt_rgb_wirh_brdf), axis=1)

            normal_map = np.concatenate((normal_rgb_map, gt_normals_rgb_map, normals_diff_map, normals_orientation_loss_map), axis=1)
            brdf_map = np.concatenate((albedo_map, roughness_map, fresnel_map), axis=1)
            single_aligned_albedo_gamma = ((single_aligned_albedo_map.numpy())**(1/2.2) * 255).astype('uint8')
            three_aligned_albedo_gamma = ((three_aligned_albedo_map.numpy())**(1/2.2) * 255).astype('uint8')
            gt_albedo_gamma = ((gt_albedo_reshaped.numpy())**(1/2.2) * 255).astype('uint8')
            albedo_map = np.concatenate((single_aligned_albedo_gamma, three_aligned_albedo_gamma, gt_albedo_gamma), axis=1)
            imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/nvs_with_brdf/{prtx}{idx:03d}.png', rgb_with_brdf_map)
            imageio.imwrite(f'{savePath}/normal/{prtx}{idx:03d}.png', normal_map)
            imageio.imwrite(f'{savePath}/normal_vis/{prtx}{idx:03d}.png', normal_rgb_vis_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}.png', brdf_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_albedo.png', albedo_map) 
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_roughness.png', roughness_map)
            imageio.imwrite(f'{savePath}/acc_map/{prtx}{idx:03d}.png', acc_map)


    # Randomly select a prediction to visualize
    if logger and step and not test_all:
        vis_idx = random.choice(range(len(rgb_maps)))
        vis_rgb = torch.from_numpy(rgb_maps[vis_idx])
        vis_rgb_brdf_rgb = torch.from_numpy(rgb_with_brdf_maps[vis_idx])
        vis_depth = torch.from_numpy(depth_maps[vis_idx])
        vis_rgb_gt = torch.from_numpy(gt_maps[vis_idx])
        vis_normal_rgb = torch.from_numpy(normal_rgb_maps[vis_idx])
        vis_normal_gt_rgb = torch.from_numpy(normals_rgb_gt_maps[vis_idx])
        vis_normals_diff_rgb = torch.from_numpy(normals_diff_maps[vis_idx])
        vis_normals_orientation_loss_rgb = torch.from_numpy(normals_orientation_loss_maps[vis_idx])
        vis_albedo = torch.from_numpy(albedo_maps[vis_idx])
        vis_single_aligned_albedo_gamma = torch.from_numpy((single_aligned_albedo_maps[vis_idx]* 255).astype('uint8'))
        vis_three_aligned_albedo_gamma = torch.from_numpy((three_aligned_albedo_maps[vis_idx]* 255).astype('uint8'))
        vis_gt_albedo_gamma = torch.from_numpy((gt_albedo_maps[vis_idx]* 255).astype('uint8'))
        vis_roughness = torch.from_numpy(roughness_maps[vis_idx])
        vis_fresnel = torch.from_numpy(fresnel_maps[vis_idx])
        vis_rgb_grid = torch.stack([vis_rgb, vis_rgb_brdf_rgb, vis_rgb_gt, vis_depth]).permute(0, 3, 1, 2).to(float)
        vis_normal_grid = torch.stack([vis_normal_rgb, vis_normal_gt_rgb, vis_normals_diff_rgb, vis_normals_orientation_loss_rgb]).permute(0, 3, 1, 2).to(float)
        vis_brdf_grid = torch.stack([vis_albedo, vis_roughness, vis_fresnel]).permute(0, 3, 1, 2).to(float)
        vis_envir_map_grid = torch.from_numpy(envirmap).unsqueeze(0).permute(0, 3, 1, 2).to(float)
        vis_albedo_grid = torch.stack([vis_single_aligned_albedo_gamma, vis_three_aligned_albedo_gamma, vis_gt_albedo_gamma]).permute(0, 3, 1, 2).to(float)


        logger.add_image('test/rgb',
                            vutils.make_grid(vis_rgb_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/normal',
                            vutils.make_grid(vis_normal_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/brdf',
                            vutils.make_grid(vis_brdf_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/envir_map',
                            vutils.make_grid(vis_envir_map_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/albedo',
                            vutils.make_grid(vis_albedo_grid, padding=0, normalize=True, value_range=(0, 255)), step)


    # Compute metrics
    if PSNRs_rgb:
        psnr = np.mean(np.asarray(PSNRs_rgb))
        psnr_rgb_brdf = np.mean(np.asarray(PSNRs_rgb_brdf))
        gt_normal_stack = np.stack(normal_gt_list)
        render_normal_stack = np.stack(normal_raw_list)

        single_aligned_albedo_maps = np.stack(single_aligned_albedo_maps)
        three_aligned_albedo_maps = np.stack(three_aligned_albedo_maps)
        gt_albedo_maps = np.stack(gt_albedo_maps)
        loss_albedo_single = np.mean((gt_albedo_maps - single_aligned_albedo_maps)**2)
        loss_albedo_three = np.mean((gt_albedo_maps - three_aligned_albedo_maps)**2)
        PSNR_albedo_single = -10.0 * np.log(loss_albedo_single) / np.log(10.0)
        PSNR_albedo_three = -10.0 * np.log(loss_albedo_three) / np.log(10.0)
        # compute mean angular error
        MAE = np.mean(np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1)) * 180 / np.pi)
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))

            ssim_rgb_brdf = np.mean(np.asarray(ssims_rgb_brdf))
            l_a_rgb_brdf = np.mean(np.asarray(l_alex_rgb_brdf))
            l_v_rgb_brdf = np.mean(np.asarray(l_vgg_rgb_brdf))

            ssim_albedo_single = np.mean(np.asarray(ssims_albedo_single))
            l_a_albedo_single = np.mean(np.asarray(l_alex_albedo_single))
            l_v_albedo_single = np.mean(np.asarray(l_vgg_albedo_single))

            ssim_albedo_three = np.mean(np.asarray(ssims_albedo_three))
            l_a_albedo_three = np.mean(np.asarray(l_alex_albedo_three))
            l_v_albedo_three = np.mean(np.asarray(l_vgg_albedo_three))


            saved_message = f'Iteration:{prtx[:-1]}: \n' \
                            + f'\tPSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}, PNSR_albedo_single_aligned: {PSNR_albedo_single:.2f}, PNSR_albedo_three_aligned: {PSNR_albedo_three:.2f}\n' \
                            + f'\tSSIM_rgb: {ssim:.4f}, L_Alex_rgb: {l_a:.4f}, L_VGG_rgb: {l_v:.4f}\n' \
                            + f'\tSSIM_rgb_brdf: {ssim_rgb_brdf:.4f}, L_Alex_rgb_brdf: {l_a_rgb_brdf:.4f}, L_VGG_rgb_brdf: {l_v_rgb_brdf:.4f}\n' \
                            + f'\tSSIM_albedo_single: {ssim_albedo_single:.4f}, L_Alex_albedo_single: {l_a_albedo_single:.4f}, L_VGG_albedo_single: {l_v_albedo_single:.4f}\n' \
                            + f'\tSSIM_albedo_three: {ssim_albedo_three:.4f}, L_Alex_albedo_three: {l_a_albedo_three:.4f}, L_VGG_albedo_three: {l_v_albedo_three:.4f}\n' \
                            + f'\tMAE: {MAE:.2f}\n'

        else:
            saved_message = f'Iteration:{prtx[:-1]}, PSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}, MAE: {MAE:.2f}, PSNR_albedo_single_aligned: {PSNR_albedo_single:.2f}, PSNR_albedo_three_aligned: {PSNR_albedo_three:.2f}\n'
        # write the end of record file
        with open(f'{savePath}/metrics_record.txt', 'a') as f:
            f.write(saved_message)

    # save video results
    if test_all:
        os.makedirs(savePath + "/video", exist_ok=True)
        video_path = savePath + "/video"
        imageio.mimsave(os.path.join(video_path, 'rgb.mp4'), np.stack(rgb_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'rgb_brdf.mp4'), np.stack(rgb_with_brdf_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'gt_normal_video.mp4'), np.stack(normals_rgb_gt_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'render_normal_video.mp4'), np.stack(normal_rgb_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'render_normal_vis_video.mp4'), np.stack(normal_rgb_vis_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'single_aligned_albedo.mp4'), (single_aligned_albedo_maps * 255).astype('uint8'), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'three_aligned_albedo.mp4'), (three_aligned_albedo_maps * 255).astype('uint8'), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'roughness.mp4'), np.stack(roughness_maps), fps=24, quality=8)

    return psnr, psnr_rgb_brdf, MAE, PSNR_albedo_single, PSNR_albedo_three



@torch.no_grad()
def evaluation_iter_TensoIR_simple(
        test_dataset,
        tensoIR,
        args,
        renderer,
        savePath=None,
        prtx='',
        N_samples=-1,
        white_bg=False,
        ndc_ray=False,
        compute_extra_metrics=True,
        device='cuda',
        logger=None,
        step=None,
        test_all=False,
):

    
    PSNRs_rgb, rgb_maps, depth_maps, gt_maps, gt_rgb_brdf_maps = [], [], [], [], []
    PSNRs_rgb_brdf = []
    rgb_with_brdf_maps, normal_rgb_maps, normal_rgb_vis_maps = [], [], []
    albedo_maps, albedo_gamma_maps, roughness_maps, fresnel_maps, normals_diff_maps, normals_orientation_loss_maps = [], [], [], [], [], []
    ssims, l_alex, l_vgg = [], [], []
    ssims_rgb_brdf, l_alex_rgb_brdf, l_vgg_rgb_brdf = [], [], []


    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/nvs_with_radiance_field", exist_ok=True)
    os.makedirs(savePath + "/nvs_with_brdf", exist_ok=True)
    os.makedirs(savePath + "/normal", exist_ok=True)
    os.makedirs(savePath + "/normal_vis", exist_ok=True)
    os.makedirs(savePath + "/brdf", exist_ok=True)
    os.makedirs(savePath + "/envir_map/", exist_ok=True)
    os.makedirs(savePath + "/acc_map", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh

    num_test = len(test_dataset) if test_all else min(args.N_vis, len(test_dataset))



    _, view_dirs = tensoIR.generate_envir_map_dir(256, 512)

    predicted_envir_map = tensoIR.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
    predicted_envir_map = predicted_envir_map.reshape(256, 512, 3).cpu().detach().numpy()
    predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
    predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
    envirmap = predicted_envir_map

    # save predicted envir map
    imageio.imwrite(f'{savePath}/envir_map/{prtx}envirmap.png', envirmap)
    test_duration = int(len(test_dataset) / num_test)


    for idx in range(num_test):
        item = test_dataset.__getitem__(idx * test_duration)
        rays = item['rays']                 # [H*W, 6]
        gt_rgb = item['rgbs'][0]            # [H*W, 3]
        light_idx = item['light_idx'][0]    # [H*W, 1]
        gt_rgb_wirh_brdf = gt_rgb           # [H*W, 3]
        gt_mask = item['rgbs_mask']         # [H*W, 1]

        rgb_map, acc_map, depth_map, normal_map, albedo_map, roughness_map, albedo_gamma_map = [], [], [], [], [], [], []
        fresnel_map, rgb_with_brdf_map, normals_diff_map, normals_orientation_loss_map = [], [], [], []
        
        chunk_idxs = torch.split(torch.arange(rays.shape[0]), args.batch_size_test)
        for chunk_idx in chunk_idxs:
            ret_kw= renderer(   
                                rays[chunk_idx], 
                                None, # not used
                                light_idx[chunk_idx],
                                tensoIR, 
                                N_samples=N_samples,
                                ndc_ray=ndc_ray,
                                white_bg=white_bg,
                                sample_method='fixed_envirmap',
                                chunk_size=args.relight_chunk_size,  
                                device=device,
                                args=args
                            )
            rgb_map.append(ret_kw['rgb_map'].detach().cpu())
            depth_map.append(ret_kw['depth_map'].detach().cpu())
            normal_map.append(ret_kw['normal_map'].detach().cpu())
            albedo_map.append(ret_kw['albedo_map'].detach().cpu())
            roughness_map.append(ret_kw['roughness_map'].detach().cpu())
            fresnel_map.append(ret_kw['fresnel_map'].detach().cpu())
            rgb_with_brdf_map.append(ret_kw['rgb_with_brdf_map'].detach().cpu())
            normals_diff_map.append(ret_kw['normals_diff_map'].detach().cpu())
            normals_orientation_loss_map.append(ret_kw['normals_orientation_loss_map'].detach().cpu())
            acc_map.append(ret_kw['acc_map'].detach().cpu())

        
        rgb_map = torch.cat(rgb_map)
        depth_map = torch.cat(depth_map)
        normal_map = torch.cat(normal_map)
        albedo_map = torch.cat(albedo_map)
        roughness_map = torch.cat(roughness_map)
        fresnel_map = torch.cat(fresnel_map)
        rgb_with_brdf_map = torch.cat(rgb_with_brdf_map)
        normals_diff_map = torch.cat(normals_diff_map)
        normals_orientation_loss_map = torch.cat(normals_orientation_loss_map)
        acc_map = torch.cat(acc_map)
        

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_with_brdf_map = rgb_with_brdf_map.clamp(0.0, 1.0)

        acc_map = acc_map.reshape(H, W).detach().cpu()

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).detach().cpu(), depth_map.reshape(H, W).detach().cpu()
        rgb_with_brdf_map = rgb_with_brdf_map.reshape(H, W, 3).detach().cpu()
        albedo_map = albedo_map.reshape(H, W, 3).detach().cpu()

        albedo_gamma_map = (albedo_map.clip(0, 1.)) ** (1.0 / 2.2)


        roughness_map = roughness_map.reshape(H, W, 1).repeat(1, 1, 3).detach().cpu()
        fresnel_map = fresnel_map.reshape(H, W, 3).detach().cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        # Store loss and images
        if test_dataset.__len__():
            gt_rgb = gt_rgb.view(H, W, 3)
            gt_rgb_wirh_brdf = gt_rgb_wirh_brdf.view(H, W, 3)
            loss_rgb = torch.mean((rgb_map - gt_rgb) ** 2)
            loss_rgb_brdf = torch.mean((rgb_with_brdf_map - gt_rgb_wirh_brdf) ** 2)
            PSNRs_rgb.append(-10.0 * np.log(loss_rgb.item()) / np.log(10.0))
            PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensoIR.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensoIR.device)

                ssim_rgb_brdf = rgb_ssim(rgb_with_brdf_map, gt_rgb_wirh_brdf, 1)
                l_a_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'alex', tensoIR.device)
                l_v_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'vgg', tensoIR.device)

                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

                ssims_rgb_brdf.append(ssim_rgb_brdf)
                l_alex_rgb_brdf.append(l_a_rgb_brdf)
                l_vgg_rgb_brdf.append(l_v_rgb_brdf)



        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_with_brdf_map = (rgb_with_brdf_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        gt_rgb_wirh_brdf = (gt_rgb_wirh_brdf.numpy() * 255).astype('uint8')
        albedo_map = (albedo_map.numpy() * 255).astype('uint8')
        albedo_gamma_map = (albedo_gamma_map.numpy() * 255).astype('uint8')
        roughness_map = (roughness_map.numpy() * 255).astype('uint8')
        fresnel_map = (fresnel_map.numpy() * 255).astype('uint8')
        acc_map = (acc_map.numpy() * 255).astype('uint8')

        # Visualize normal
        ## Prediction
        normal_map = F.normalize(normal_map, dim=-1)

        normal_rgb_map = normal_map * 0.5 + 0.5 # map from [-1, 1] to [0, 1] to visualize
        normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype('uint8')
        normal_rgb_vis_map = (normal_rgb_map * (acc_map[:, :, None] / 255.0) + (1 -(acc_map[:, :, None] / 255.0)) * 255).astype('uint8') # white background


        # difference between the predicted normals and derived normals
        normals_diff_map = (torch.clamp(normals_diff_map, 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        # normals orientation loss map
        normals_orientation_loss_map = (torch.clamp(normals_orientation_loss_map , 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        rgb_with_brdf_maps.append(rgb_with_brdf_map)
        depth_maps.append(depth_map)
        gt_maps.append(gt_rgb)
        gt_rgb_brdf_maps.append(gt_rgb_wirh_brdf)
        normal_rgb_maps.append(normal_rgb_map)
        normal_rgb_vis_maps.append(normal_rgb_vis_map)

        if not test_all:
            normals_diff_maps.append(normals_diff_map)
            normals_orientation_loss_maps.append(normals_orientation_loss_map)


        albedo_maps.append(albedo_map)
        albedo_gamma_maps.append(albedo_gamma_map)
        roughness_maps.append(roughness_map)
        fresnel_maps.append(fresnel_map)


        if savePath is not None:
            rgb_map = np.concatenate((rgb_map, gt_rgb, depth_map), axis=1)
            rgb_with_brdf_map = np.concatenate((rgb_with_brdf_map, gt_rgb_wirh_brdf), axis=1)

            normal_map = np.concatenate((normal_rgb_map, normals_diff_map, normals_orientation_loss_map), axis=1)
            brdf_map = np.concatenate((albedo_map, roughness_map, fresnel_map), axis=1)
            
            imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/nvs_with_brdf/{prtx}{idx:03d}.png', rgb_with_brdf_map)
            imageio.imwrite(f'{savePath}/normal/{prtx}{idx:03d}.png', normal_map)
            imageio.imwrite(f'{savePath}/normal_vis/{prtx}{idx:03d}.png', normal_rgb_vis_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}.png', brdf_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_albedo.png', albedo_gamma_map) 
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_roughness.png', roughness_map)
            imageio.imwrite(f'{savePath}/acc_map/{prtx}{idx:03d}.png', acc_map)


    # Randomly select a prediction to visualize
    if logger and step and not test_all:
        vis_idx = random.choice(range(len(rgb_maps)))
        vis_rgb = torch.from_numpy(rgb_maps[vis_idx])
        vis_rgb_brdf_rgb = torch.from_numpy(rgb_with_brdf_maps[vis_idx])
        vis_depth = torch.from_numpy(depth_maps[vis_idx])
        vis_rgb_gt = torch.from_numpy(gt_maps[vis_idx])
        vis_normal_rgb = torch.from_numpy(normal_rgb_maps[vis_idx])
        vis_normals_diff_rgb = torch.from_numpy(normals_diff_maps[vis_idx])
        vis_normals_orientation_loss_rgb = torch.from_numpy(normals_orientation_loss_maps[vis_idx])
        vis_albedo = torch.from_numpy(albedo_maps[vis_idx])
        vis_albedo_gamma = torch.from_numpy(albedo_gamma_maps[vis_idx])
        vis_roughness = torch.from_numpy(roughness_maps[vis_idx])
        vis_fresnel = torch.from_numpy(fresnel_maps[vis_idx])
        vis_rgb_grid = torch.stack([vis_rgb, vis_rgb_brdf_rgb, vis_rgb_gt, vis_depth]).permute(0, 3, 1, 2).to(float)
        vis_normal_grid = torch.stack([vis_normal_rgb, vis_normals_diff_rgb, vis_normals_orientation_loss_rgb]).permute(0, 3, 1, 2).to(float)
        vis_brdf_grid = torch.stack([vis_albedo, vis_roughness, vis_fresnel]).permute(0, 3, 1, 2).to(float)
        vis_envir_map_grid = torch.from_numpy(envirmap).unsqueeze(0).permute(0, 3, 1, 2).to(float)
        vis_albedo_grid = torch.stack([vis_albedo, vis_albedo_gamma]).permute(0, 3, 1, 2).to(float)


        logger.add_image('test/rgb',
                            vutils.make_grid(vis_rgb_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/normal',
                            vutils.make_grid(vis_normal_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/brdf',
                            vutils.make_grid(vis_brdf_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/envir_map',
                            vutils.make_grid(vis_envir_map_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/albedo',
                            vutils.make_grid(vis_albedo_grid, padding=0, normalize=True, value_range=(0, 255)), step)


    # Compute metrics
    if PSNRs_rgb:
        psnr = np.mean(np.asarray(PSNRs_rgb))
        psnr_rgb_brdf = np.mean(np.asarray(PSNRs_rgb_brdf))
   

        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))

            ssim_rgb_brdf = np.mean(np.asarray(ssims_rgb_brdf))
            l_a_rgb_brdf = np.mean(np.asarray(l_alex_rgb_brdf))
            l_v_rgb_brdf = np.mean(np.asarray(l_vgg_rgb_brdf))



            saved_message = f'Iteration:{prtx[:-1]}: \n' \
                            + f'\tPSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}\n' \
                            + f'\tSSIM_rgb: {ssim:.4f}, L_Alex_rgb: {l_a:.4f}, L_VGG_rgb: {l_v:.4f}\n' \
                            + f'\tSSIM_rgb_brdf: {ssim_rgb_brdf:.4f}, L_Alex_rgb_brdf: {l_a_rgb_brdf:.4f}, L_VGG_rgb_brdf: {l_v_rgb_brdf:.4f}\n' 

        else:
            saved_message = f'Iteration:{prtx[:-1]}, PSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}\n'
        # write the end of record file
        with open(f'{savePath}/metrics_record.txt', 'a') as f:
            f.write(saved_message)

    return psnr, psnr_rgb_brdf



@torch.no_grad()
def evaluation_iter_TensoIR_general_multi_lights(
        test_dataset,
        tensoIR,
        args,
        renderer,
        savePath=None,
        prtx='',
        N_samples=-1,
        white_bg=False,
        ndc_ray=False,
        compute_extra_metrics=True,
        device='cuda',
        logger=None,
        step=None,
        test_all=False,
        light_idx_to_test=-1,
):

    
    PSNRs_rgb, rgb_maps, depth_maps, gt_maps, gt_rgb_brdf_maps = [], [], [], [], []
    PSNRs_rgb_brdf = []
    rgb_with_brdf_maps, normal_rgb_maps, normal_rgb_vis_maps, normals_rgb_gt_maps = [], [], [], []
    albedo_maps, single_aligned_albedo_maps, three_aligned_albedo_maps, gt_albedo_maps, roughness_maps, fresnel_maps, normals_diff_maps, normals_orientation_loss_maps  = [], [], [], [], [], [], [], []
    normal_raw_list = []
    normal_gt_list = []
    ssims, l_alex, l_vgg = [], [], []
    ssims_rgb_brdf, l_alex_rgb_brdf, l_vgg_rgb_brdf = [], [], []
    ssims_albedo_single, l_alex_albedo_single, l_vgg_albedo_single = [], [], []
    ssims_albedo_three, l_alex_albedo_three, l_vgg_albedo_three = [], [], []

    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/nvs_with_radiance_field", exist_ok=True)
    os.makedirs(savePath + "/nvs_with_brdf", exist_ok=True)
    os.makedirs(savePath + "/normal", exist_ok=True)
    os.makedirs(savePath + "/normal_vis", exist_ok=True)
    os.makedirs(savePath + "/brdf", exist_ok=True)
    os.makedirs(savePath + "/envir_map/", exist_ok=True)
    os.makedirs(savePath + "/acc_map", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh

    num_test = len(test_dataset) if test_all else min(args.N_vis, len(test_dataset))

    gt_envir_map = None
    # if test_dataset.lights_probes is not None:
    #     gt_envir_map = test_dataset.lights_probes.reshape(test_dataset.envir_map_h, test_dataset.envir_map_w, 3).numpy()
    #     gt_envir_map = np.uint8(np.clip(np.power(gt_envir_map, 1./2.2), 0., 1.) * 255.)
        
    #     # resize to 256 * 512
    #     gt_envir_map = cv2.resize(gt_envir_map, (512, 256), interpolation=cv2.INTER_CUBIC)

    _, view_dirs = tensoIR.generate_envir_map_dir(256, 512)

    predicted_envir_map = tensoIR.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))
    predicted_envir_map = predicted_envir_map.reshape(256 * tensoIR.light_num, 512, 3).cpu().detach().numpy()
    predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
    predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
    if gt_envir_map is not None:
        envirmap = np.concatenate((gt_envir_map, predicted_envir_map), axis=1)
    else:
        envirmap = predicted_envir_map
    # save predicted envir map
    imageio.imwrite(f'{savePath}/envir_map/{prtx}envirmap.png', envirmap)
    test_duration = int(len(test_dataset) / num_test)

    # compute global rescale ratio for predicted albedo
    if test_all:
        global_rescale_value_single, global_rescale_value_three = compute_rescale_ratio(tensoIR, test_dataset, sampled_num=20)
        global_rescale_value_single, global_rescale_value_three = global_rescale_value_single.cpu(), global_rescale_value_three.cpu()


    for idx in range(num_test):
        item = test_dataset.__getitem__(idx * test_duration)
        # generate a random number between [0, tensoIR.light_num)
        if light_idx_to_test >= 0:
            light_kind_idx = light_idx_to_test
        else:
            light_kind_idx = int(np.random.randint(tensoIR.light_num))
        rays = item['rays']                 # [H*W, 6]
        gt_rgb = item['rgbs'][light_kind_idx]    # [H*W, 3]
        light_idx = item['light_idx'][light_kind_idx]        # [H*W]
        gt_normals = item['normals']        # [H*W, 3]
        gt_rgb_wirh_brdf = gt_rgb           # [H*W, 3]
        gt_mask = item['rgbs_mask']         # [H*W, 1]
        albedo_gt = item['albedo']          # [H*W, 3]
        rgb_map, acc_map, depth_map, normal_map, albedo_map, roughness_map = [], [], [], [], [], []
        fresnel_map, rgb_with_brdf_map, normals_diff_map, normals_orientation_loss_map = [], [], [], []

        
        chunk_idxs = torch.split(torch.arange(rays.shape[0]), args.batch_size_test)
        for chunk_idx in chunk_idxs:
            ret_kw= renderer(   
                                rays[chunk_idx], 
                                None, # not used
                                light_idx[chunk_idx],
                                tensoIR, 
                                N_samples=N_samples,
                                ndc_ray=ndc_ray,
                                white_bg=white_bg,
                                sample_method='fixed_envirmap',
                                chunk_size=args.relight_chunk_size,  
                                device=device,
                                args=args
                            )
            rgb_map.append(ret_kw['rgb_map'].detach().cpu())
            depth_map.append(ret_kw['depth_map'].detach().cpu())
            normal_map.append(ret_kw['normal_map'].detach().cpu())
            albedo_map.append(ret_kw['albedo_map'].detach().cpu())
            roughness_map.append(ret_kw['roughness_map'].detach().cpu())
            fresnel_map.append(ret_kw['fresnel_map'].detach().cpu())
            rgb_with_brdf_map.append(ret_kw['rgb_with_brdf_map'].detach().cpu())
            normals_diff_map.append(ret_kw['normals_diff_map'].detach().cpu())
            normals_orientation_loss_map.append(ret_kw['normals_orientation_loss_map'].detach().cpu())
            acc_map.append(ret_kw['acc_map'].detach().cpu())

        
        rgb_map = torch.cat(rgb_map)
        depth_map = torch.cat(depth_map)
        normal_map = torch.cat(normal_map)
        albedo_map = torch.cat(albedo_map)
        roughness_map = torch.cat(roughness_map)
        fresnel_map = torch.cat(fresnel_map)
        rgb_with_brdf_map = torch.cat(rgb_with_brdf_map)
        normals_diff_map = torch.cat(normals_diff_map)
        normals_orientation_loss_map = torch.cat(normals_orientation_loss_map)
        acc_map = torch.cat(acc_map)

        # normal_map_to_test = acc_map[..., None] * normal_map + (1 - acc_map[..., None]) * torch.tensor([0.0, 0.0, 1.0])

        

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_with_brdf_map = rgb_with_brdf_map.clamp(0.0, 1.0)

        acc_map = acc_map.reshape(H, W).detach().cpu()

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).detach().cpu(), depth_map.reshape(H, W).detach().cpu()
        rgb_with_brdf_map = rgb_with_brdf_map.reshape(H, W, 3).detach().cpu()
        albedo_map = albedo_map.reshape(H, W, 3).detach().cpu()

        single_aligned_albedo_map = torch.ones_like(albedo_map)
        three_aligned_albedo_map = torch.ones_like(albedo_map)
        gt_albedo_reshaped = albedo_gt.reshape(H, W, 3).detach().cpu()
        gt_mask_reshaped = gt_mask.reshape(H, W).detach().cpu()
        # single channel alignment for albedo
        if test_all:
            ratio_value = global_rescale_value_single
        else:
            ratio_value = (gt_albedo_reshaped[gt_mask_reshaped] / albedo_map[gt_mask_reshaped].clamp(min=1e-6))[..., 0].median()
        single_aligned_albedo_map[gt_mask_reshaped] = (ratio_value * albedo_map[gt_mask_reshaped]).clamp(min=0.0, max=1.0)
        # three channel alignment for albedo
        if test_all:
            ratio_value = global_rescale_value_three
        else:
            ratio_value, _ = (gt_albedo_reshaped[gt_mask_reshaped]/ albedo_map[gt_mask_reshaped].clamp(min=1e-6)).median(dim=0)
        three_aligned_albedo_map[gt_mask_reshaped] = (ratio_value * albedo_map[gt_mask_reshaped]).clamp(min=0.0, max=1.0)


        roughness_map = roughness_map.reshape(H, W, 1).repeat(1, 1, 3).detach().cpu()
        fresnel_map = fresnel_map.reshape(H, W, 3).detach().cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        # Store loss and images
        if test_dataset.__len__():
            gt_rgb = gt_rgb.view(H, W, 3)
            gt_rgb_wirh_brdf = gt_rgb_wirh_brdf.view(H, W, 3)
            loss_rgb = torch.mean((rgb_map - gt_rgb) ** 2)
            loss_rgb_brdf = torch.mean((rgb_with_brdf_map - gt_rgb_wirh_brdf) ** 2)
            PSNRs_rgb.append(-10.0 * np.log(loss_rgb.item()) / np.log(10.0))
            PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensoIR.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensoIR.device)

                ssim_rgb_brdf = rgb_ssim(rgb_with_brdf_map, gt_rgb_wirh_brdf, 1)
                l_a_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'alex', tensoIR.device)
                l_v_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'vgg', tensoIR.device)

                # single channel aligned albedo
                ssim_albedo_single = rgb_ssim(single_aligned_albedo_map, gt_albedo_reshaped, 1)
                l_a_albedo_single = rgb_lpips(gt_albedo_reshaped.numpy(), single_aligned_albedo_map.numpy(), 'alex', tensoIR.device)
                l_v_albedo_single = rgb_lpips(gt_albedo_reshaped.numpy(), single_aligned_albedo_map.numpy(), 'vgg', tensoIR.device)
                # three channel aligned albedo
                ssim_albedo_three = rgb_ssim(three_aligned_albedo_map, gt_albedo_reshaped, 1)
                l_a_albedo_three = rgb_lpips(gt_albedo_reshaped.numpy(), three_aligned_albedo_map.numpy(), 'alex', tensoIR.device)
                l_v_albedo_three = rgb_lpips(gt_albedo_reshaped.numpy(), three_aligned_albedo_map.numpy(), 'vgg', tensoIR.device)

                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

                ssims_rgb_brdf.append(ssim_rgb_brdf)
                l_alex_rgb_brdf.append(l_a_rgb_brdf)
                l_vgg_rgb_brdf.append(l_v_rgb_brdf)

                ssims_albedo_single.append(ssim_albedo_single)
                l_alex_albedo_single.append(l_a_albedo_single)
                l_vgg_albedo_single.append(l_v_albedo_single)

                ssims_albedo_three.append(ssim_albedo_three)
                l_alex_albedo_three.append(l_a_albedo_three)
                l_vgg_albedo_three.append(l_v_albedo_three)




        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_with_brdf_map = (rgb_with_brdf_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        gt_rgb_wirh_brdf = (gt_rgb_wirh_brdf.numpy() * 255).astype('uint8')
        albedo_map = (albedo_map.numpy() * 255).astype('uint8')
        roughness_map = (roughness_map.numpy() * 255).astype('uint8')
        fresnel_map = (fresnel_map.numpy() * 255).astype('uint8')
        acc_map = (acc_map.numpy() * 255).astype('uint8')

        # Visualize normal
        ## Prediction
        normal_map = F.normalize(normal_map, dim=-1)

        normal_raw_list.append(normal_map)

        normal_rgb_map = normal_map * 0.5 + 0.5 # map from [-1, 1] to [0, 1] to visualize
        normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype('uint8')
        normal_rgb_vis_map = (normal_rgb_map * (acc_map[:, :, None] / 255.0) + (1 -(acc_map[:, :, None] / 255.0)) * 255).astype('uint8') # white background


        # GT normal 
        gt_normals = F.normalize(gt_normals, dim=-1)
        normal_gt_list.append(gt_normals)
        gt_normals_rgb_map = gt_normals * 0.5 + 0.5
        gt_normals_rgb_map = (gt_normals_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')

        # difference between the predicted normals and derived normals
        normals_diff_map = (torch.clamp(normals_diff_map, 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        # normals orientation loss map
        normals_orientation_loss_map = (torch.clamp(normals_orientation_loss_map , 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        rgb_with_brdf_maps.append(rgb_with_brdf_map)
        depth_maps.append(depth_map)
        gt_maps.append(gt_rgb)
        gt_rgb_brdf_maps.append(gt_rgb_wirh_brdf)
        normal_rgb_maps.append(normal_rgb_map)
        normal_rgb_vis_maps.append(normal_rgb_vis_map)
        normals_rgb_gt_maps.append(gt_normals_rgb_map)
        if not test_all:
            normals_diff_maps.append(normals_diff_map)
            normals_orientation_loss_maps.append(normals_orientation_loss_map)


        albedo_maps.append(albedo_map)
        single_aligned_albedo_maps.append((single_aligned_albedo_map.numpy())**(1/2.2))
        three_aligned_albedo_maps.append((three_aligned_albedo_map.numpy())**(1/2.2))
        gt_albedo_maps.append((gt_albedo_reshaped.numpy())**(1/2.2))
        roughness_maps.append(roughness_map)
        fresnel_maps.append(fresnel_map)


        if savePath is not None:
            rgb_map = np.concatenate((rgb_map, gt_rgb, depth_map), axis=1)
            rgb_with_brdf_map = np.concatenate((rgb_with_brdf_map, gt_rgb_wirh_brdf), axis=1)

            normal_map = np.concatenate((normal_rgb_map, gt_normals_rgb_map, normals_diff_map, normals_orientation_loss_map), axis=1)
            brdf_map = np.concatenate((albedo_map, roughness_map, fresnel_map), axis=1)
            single_aligned_albedo_gamma = ((single_aligned_albedo_map.numpy())**(1/2.2) * 255).astype('uint8')
            three_aligned_albedo_gamma = ((three_aligned_albedo_map.numpy())**(1/2.2) * 255).astype('uint8')
            gt_albedo_gamma = ((gt_albedo_reshaped.numpy())**(1/2.2) * 255).astype('uint8')
            albedo_map = np.concatenate((single_aligned_albedo_gamma, three_aligned_albedo_gamma, gt_albedo_gamma), axis=1)
            imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/nvs_with_brdf/{prtx}{idx:03d}.png', rgb_with_brdf_map)
            imageio.imwrite(f'{savePath}/normal/{prtx}{idx:03d}.png', normal_map)
            imageio.imwrite(f'{savePath}/normal_vis/{prtx}{idx:03d}.png', normal_rgb_vis_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}.png', brdf_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_albedo.png', albedo_map) 
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_roughness.png', roughness_map)
            imageio.imwrite(f'{savePath}/acc_map/{prtx}{idx:03d}.png', acc_map)


    # Randomly select a prediction to visualize
    if logger and step and not test_all:
        vis_idx = random.choice(range(len(rgb_maps)))
        vis_rgb = torch.from_numpy(rgb_maps[vis_idx])
        vis_rgb_brdf_rgb = torch.from_numpy(rgb_with_brdf_maps[vis_idx])
        vis_depth = torch.from_numpy(depth_maps[vis_idx])
        vis_rgb_gt = torch.from_numpy(gt_maps[vis_idx])
        vis_normal_rgb = torch.from_numpy(normal_rgb_maps[vis_idx])
        vis_normal_gt_rgb = torch.from_numpy(normals_rgb_gt_maps[vis_idx])
        vis_normals_diff_rgb = torch.from_numpy(normals_diff_maps[vis_idx])
        vis_normals_orientation_loss_rgb = torch.from_numpy(normals_orientation_loss_maps[vis_idx])
        vis_albedo = torch.from_numpy(albedo_maps[vis_idx])
        vis_single_aligned_albedo_gamma = torch.from_numpy((single_aligned_albedo_maps[vis_idx]* 255).astype('uint8'))
        vis_three_aligned_albedo_gamma = torch.from_numpy((three_aligned_albedo_maps[vis_idx]* 255).astype('uint8'))
        vis_gt_albedo_gamma = torch.from_numpy((gt_albedo_maps[vis_idx]* 255).astype('uint8'))
        vis_roughness = torch.from_numpy(roughness_maps[vis_idx])
        vis_fresnel = torch.from_numpy(fresnel_maps[vis_idx])
        vis_rgb_grid = torch.stack([vis_rgb, vis_rgb_brdf_rgb, vis_rgb_gt, vis_depth]).permute(0, 3, 1, 2).to(float)
        vis_normal_grid = torch.stack([vis_normal_rgb, vis_normal_gt_rgb, vis_normals_diff_rgb, vis_normals_orientation_loss_rgb]).permute(0, 3, 1, 2).to(float)
        vis_brdf_grid = torch.stack([vis_albedo, vis_roughness, vis_fresnel]).permute(0, 3, 1, 2).to(float)
        vis_envir_map_grid = torch.from_numpy(envirmap).unsqueeze(0).permute(0, 3, 1, 2).to(float)
        vis_albedo_grid = torch.stack([vis_single_aligned_albedo_gamma, vis_three_aligned_albedo_gamma, vis_gt_albedo_gamma]).permute(0, 3, 1, 2).to(float)


        logger.add_image('test/rgb',
                            vutils.make_grid(vis_rgb_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/normal',
                            vutils.make_grid(vis_normal_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/brdf',
                            vutils.make_grid(vis_brdf_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/envir_map',
                            vutils.make_grid(vis_envir_map_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/albedo',
                            vutils.make_grid(vis_albedo_grid, padding=0, normalize=True, value_range=(0, 255)), step)


    # Compute metrics
    if PSNRs_rgb:
        psnr = np.mean(np.asarray(PSNRs_rgb))
        psnr_rgb_brdf = np.mean(np.asarray(PSNRs_rgb_brdf))
        gt_normal_stack = np.stack(normal_gt_list)
        render_normal_stack = np.stack(normal_raw_list)

        single_aligned_albedo_maps = np.stack(single_aligned_albedo_maps)
        three_aligned_albedo_maps = np.stack(three_aligned_albedo_maps)
        gt_albedo_maps = np.stack(gt_albedo_maps)
        loss_albedo_single = np.mean((gt_albedo_maps - single_aligned_albedo_maps)**2)
        loss_albedo_three = np.mean((gt_albedo_maps - three_aligned_albedo_maps)**2)
        PSNR_albedo_single = -10.0 * np.log(loss_albedo_single) / np.log(10.0)
        PSNR_albedo_three = -10.0 * np.log(loss_albedo_three) / np.log(10.0)
        # compute mean angular error
        MAE = np.mean(np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1)) * 180 / np.pi)
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))

            ssim_rgb_brdf = np.mean(np.asarray(ssims_rgb_brdf))
            l_a_rgb_brdf = np.mean(np.asarray(l_alex_rgb_brdf))
            l_v_rgb_brdf = np.mean(np.asarray(l_vgg_rgb_brdf))

            ssim_albedo_single = np.mean(np.asarray(ssims_albedo_single))
            l_a_albedo_single = np.mean(np.asarray(l_alex_albedo_single))
            l_v_albedo_single = np.mean(np.asarray(l_vgg_albedo_single))

            ssim_albedo_three = np.mean(np.asarray(ssims_albedo_three))
            l_a_albedo_three = np.mean(np.asarray(l_alex_albedo_three))
            l_v_albedo_three = np.mean(np.asarray(l_vgg_albedo_three))


            saved_message = f'Iteration:{prtx[:-1]}: \n' \
                            + f'\tPSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}, PNSR_albedo_single_aligned: {PSNR_albedo_single:.2f}, PNSR_albedo_three_aligned: {PSNR_albedo_three:.2f}\n' \
                            + f'\tSSIM_rgb: {ssim:.4f}, L_Alex_rgb: {l_a:.4f}, L_VGG_rgb: {l_v:.4f}\n' \
                            + f'\tSSIM_rgb_brdf: {ssim_rgb_brdf:.4f}, L_Alex_rgb_brdf: {l_a_rgb_brdf:.4f}, L_VGG_rgb_brdf: {l_v_rgb_brdf:.4f}\n' \
                            + f'\tSSIM_albedo_single: {ssim_albedo_single:.4f}, L_Alex_albedo_single: {l_a_albedo_single:.4f}, L_VGG_albedo_single: {l_v_albedo_single:.4f}\n' \
                            + f'\tSSIM_albedo_three: {ssim_albedo_three:.4f}, L_Alex_albedo_three: {l_a_albedo_three:.4f}, L_VGG_albedo_three: {l_v_albedo_three:.4f}\n' \
                            + f'\tMAE: {MAE:.2f}\n'

        else:
            saved_message = f'Iteration:{prtx[:-1]}, PSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}, MAE: {MAE:.2f}, PSNR_albedo_single_aligned: {PSNR_albedo_single:.2f}, PSNR_albedo_three_aligned: {PSNR_albedo_three:.2f}\n'
        # write the end of record file
        with open(f'{savePath}/metrics_record.txt', 'a') as f:
            f.write(saved_message)

    # save video results
    if test_all:
        os.makedirs(savePath + "/video", exist_ok=True)
        video_path = savePath + "/video"
        imageio.mimsave(os.path.join(video_path, 'rgb.mp4'), np.stack(rgb_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'rgb_brdf.mp4'), np.stack(rgb_with_brdf_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'gt_normal_video.mp4'), np.stack(normals_rgb_gt_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'render_normal_video.mp4'), np.stack(normal_rgb_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'render_normal_vis_video.mp4'), np.stack(normal_rgb_vis_maps), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'single_aligned_albedo.mp4'), (single_aligned_albedo_maps * 255).astype('uint8'), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'three_aligned_albedo.mp4'), (three_aligned_albedo_maps * 255).astype('uint8'), fps=24, quality=8)
        imageio.mimsave(os.path.join(video_path, 'roughness.mp4'), np.stack(roughness_maps), fps=24, quality=8)

    return psnr, psnr_rgb_brdf, MAE, PSNR_albedo_single, PSNR_albedo_three