
import numpy as np
import cv2
from loguru import logger
import torch
import torch.nn.functional as F
from models.relight_utils import *
from models.tensoRF_init import raw2alpha
import os



def safe_l2_normalize(x, dim=None, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)


def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec

# !!!
brdf_specular = GGX_specular



def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.contiguous().view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


class Environment_Light():
    def __init__(self, hdr_path, device='cuda'):
        # transverse the hdr image to get the environment light
        files = os.listdir(hdr_path)
        self.hdr_rgbs = dict()
        self.hdr_pdf_sample = dict()
        self.hdr_pdf_return = dict()
        self.hdr_dir = dict()
        for file in files:
            if file.endswith(".hdr"):
                self.hdr_path = os.path.join(hdr_path, file)
                light_name = file.split(".")[0]
                light_rgbs = read_hdr(self.hdr_path)
                light_rgbs = torch.from_numpy(light_rgbs)
                self.hdr_rgbs[light_name] = light_rgbs.to(device)
                # compute the pdf of importance sampling of the environment map
                light_intensity = torch.sum(light_rgbs, dim=2, keepdim=True) # [H, W, 1]
                env_map_h, env_map_w, _ = light_intensity.shape
                h_interval = 1.0 / env_map_h
                sin_theta = torch.sin(torch.linspace(0 + 0.5 * h_interval, np.pi - 0.5 * h_interval, env_map_h))
                pdf = light_intensity * sin_theta.view(-1, 1, 1) # [H, W, 1]
                pdf = pdf / torch.sum(pdf)
                pdf_return = pdf * env_map_h * env_map_w / (2 * np.pi * np.pi * sin_theta.view(-1, 1, 1)) 
                self.hdr_pdf_sample[light_name] = pdf.to(device)
                self.hdr_pdf_return[light_name] = pdf_return.to(device)

                lat_step_size = np.pi / env_map_h
                lng_step_size = 2 * np.pi / env_map_w
                phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, env_map_h), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, env_map_w)], indexing='ij')


                view_dirs = torch.stack([  torch.cos(theta) * torch.cos(phi), 
                                        torch.sin(theta) * torch.cos(phi), 
                                        torch.sin(phi)], dim=-1).view(env_map_h, env_map_w, 3)    # [envH, envW, 3]
                self.hdr_dir[light_name] = view_dirs.to(device)
        self.envir_map_uniform_pdf = torch.ones_like(light_intensity) * sin_theta.view(-1, 1, 1) / (env_map_h * env_map_w)
        self.envir_map_uniform_pdf = (self.envir_map_uniform_pdf / torch.sum(self.envir_map_uniform_pdf)).to(device)
        self.envir_map_uniform_pdf_return = self.envir_map_uniform_pdf * env_map_h * env_map_w / (2 * np.pi * np.pi  * sin_theta.view(-1, 1, 1).to(device))
    @torch.no_grad()
    def sample_light(self, light_name, bs, num_samples, sample_type="importance"):
        '''
        - Args:
            - light_name: the name of the light
            - bs: batch size
            - num_samples: the number of samples
        - Returns:
            - light_dir: the direction of the light [bs, num_samples, 3]
            - light_rgb: the rgb of the light [bs, num_samples, 3]
            - light_pdf: the pdf of the light [bs, num_samples, 1]
        '''
        if sample_type == "importance":
            environment_map = self.hdr_rgbs[light_name]
            environment_map_pdf_sample = self.hdr_pdf_sample[light_name].view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
            environment_map_pdf_return = self.hdr_pdf_return[light_name].view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
            environment_map_dir = self.hdr_dir[light_name].view(-1, 3).expand(bs, -1, -1) # [bs, env_map_h * env_map_w, 3]
            environment_map_rgb = environment_map.view(-1, 3).expand(bs, -1, -1) # [bs, env_map_h * env_map_w, 3]

            # sampled the light directions
            light_dir_idx = torch.multinomial(environment_map_pdf_sample, num_samples, replacement=True) # [bs, num_samples]
            light_dir = environment_map_dir.gather(1, light_dir_idx.unsqueeze(-1).expand(-1, -1, 3)).view(bs, num_samples, 3) # [bs, num_samples, 3]
            # sampled light rgbs and pdfs
            light_rgb = environment_map_rgb.gather(1, light_dir_idx.unsqueeze(-1).expand(-1, -1, 3)).view(bs, num_samples, 3) # [bs, num_samples, 3]
            light_pdf = environment_map_pdf_return.gather(1, light_dir_idx).unsqueeze(-1) # [bs, num_samples, 1]
        elif sample_type == "uniform":
            environment_map = self.hdr_rgbs[light_name]
            environment_map_pdf_sample = self.envir_map_uniform_pdf.view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
            environment_map_pdf_return = self.envir_map_uniform_pdf_return.view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
            environment_map_dir = self.hdr_dir[light_name].view(-1, 3).expand(bs, -1, -1) # [bs, env_map_h * env_map_w, 3]
            environment_map_rgb = environment_map.view(-1, 3).expand(bs, -1, -1) # [bs, env_map_h * env_map_w, 3]

            # sampled the light directions
            light_dir_idx = torch.multinomial(environment_map_pdf_sample, num_samples, replacement=True) # [bs, num_samples]
            light_dir = environment_map_dir.gather(1, light_dir_idx.unsqueeze(-1).expand(-1, -1, 3)).view(bs, num_samples, 3) # [bs, num_samples, 3]
            # sampled light rgbs and pdfs
            light_rgb = environment_map_rgb.gather(1, light_dir_idx.unsqueeze(-1).expand(-1, -1, 3)).view(bs, num_samples, 3) # [bs, num_samples, 3]
            light_pdf = environment_map_pdf_return.gather(1, light_dir_idx).unsqueeze(-1) # [bs, num_samples, 1]
        
        return light_dir, light_rgb, light_pdf




    def get_light(self, light_name, incident_dir):

        envir_map = self.hdr_rgbs[light_name]
        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        phi = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs




def predict_visibility_by_chunk(vis_model, 
                                surface_pts, 
                                surf2light, 
                                chunk_size=40960,
                                device='cuda'):
    '''predict visibility for each point at each direction using visbility network
    - args:
        - vis_model: visibility network used to predict visibility for each point at each direction
        - surface_pts: [N, 3] surface points
        - surf2light: [N, 3], light incident direction for each surface point, pointing from surface to light
    - return:
        - visibility: [N, ] visibility for each point at each direction
    '''

    # expand the shape for pts to make it the same as light_xyz
    visibility =torch.zeros((surface_pts.shape[0], 1), dtype=torch.float32).to(device) # [N, 1]
    chunk_idxs = torch.split(torch.arange(surface_pts.shape[0]), chunk_size) # to save memory TODO: chunk size should be configurable
    for chunk_idx in chunk_idxs:
        chunk_surf2light = surf2light[chunk_idx]
        chunk_surface_pts = surface_pts[chunk_idx]
        chunk_visibility = vis_model(chunk_surface_pts, chunk_surf2light) # [N, 1]
        visibility[chunk_idx] = chunk_visibility

    return visibility

@torch.no_grad()
def get_visibility_and_indirect_light(  visibility_net, 
                                        tensoIR, 
                                        surface_pts, 
                                        surf2light, 
                                        light_idx, 
                                        nSample=96, 
                                        vis_near=0.05, 
                                        vis_far=1.5,
                                        chunk_size=40960, 
                                        device='cuda'):
    '''predict visibility for each point at each direction using visbility network
    - args:
        - visibility_net: visibility network used to predict visibility for each point at each direction
        - tensoIR: tensoIR model is used to compute the visibility and indirect lighting
        - surface_pts: [N, 3] surface points location
        - surf2light: [N, 3], light incident direction for each surface point, pointing from surface to light
        - light_idx: [N, 1], index of lighitng
        - nSample: number of samples for each ray along incident light direction
    - return:
        - visibility_predict: [N, 1] visibility result from the visibility net
        - visibility_compute: [N, 1] visibility result by choosing some directions and then computing the density
        - indirect_light: [N, 3] indirect light in the corresponding direction
        - computed_visbility_mask: [N, 1] mask indicating whether the direction is invisible to the direct light
    '''


    visibility_predict = torch.zeros((surface_pts.shape[0]), dtype=torch.float32).to(device) # [N, 1]
    visibility_compute = torch.zeros((surface_pts.shape[0]), dtype=torch.float32).to(device) # [N, 1]
    indirect_light = torch.zeros((surface_pts.shape[0], 3), dtype=torch.float32).to(device) # [N, 1]
    with torch.enable_grad():
        chunk_idxs_vis_predict = torch.split(torch.arange(surface_pts.shape[0]), 40960) # TODO: chunk size should be configurable
        # predict all directions
        for chunk_idx in chunk_idxs_vis_predict:
            chunk_surf2light = surf2light[chunk_idx]
            chunk_surface_pts = surface_pts[chunk_idx]
            chunk_visibility = visibility_net(chunk_surface_pts, chunk_surf2light) # [N, 1]
            visibility_predict[chunk_idx] = chunk_visibility.squeeze(-1) # [N, ]
        
        invisibile_to_direct_light_mask = visibility_predict < 0.5 # [N, 1] index of ray where the direct light is not visible
        visibility_predict = visibility_predict.reshape(-1, 1)  # [N, 1]

    surface_pts_masked = surface_pts[invisibile_to_direct_light_mask] # [masked(N), 3]
    surf2light_masked = surf2light[invisibile_to_direct_light_mask] # [masked(N), 3]
    light_idx_masked = light_idx[invisibile_to_direct_light_mask] # [masked(N), 1]
    visibility_masked = torch.zeros((surface_pts_masked.shape[0]), dtype=torch.float32).to(device) # [masked(N), 1]
    indirect_light_masked = torch.zeros((surface_pts_masked.shape[0], 3), dtype=torch.float32).to(device) # [masked(N), 1]
    chunk_idxs_vis_compute = torch.split(torch.arange(surface_pts_masked.shape[0]), 20480) # TODO: chunk size should be configurable
    # compute the directions where the direct light is not visible
    for chunk_idx in chunk_idxs_vis_compute:
        chunk_surface_pts = surface_pts_masked[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = surf2light_masked[chunk_idx]    # [chunk_size, 3]
        chunk_light_idx = light_idx_masked[chunk_idx]     # [chunk_size, 1]
        nerv_vis_chunk, nerfactor_vis_chunk, indirect_light_chunk = compute_radiance(tensoIR=tensoIR, 
                                                                                    surf_pts=chunk_surface_pts, 
                                                                                    light_in_dir=chunk_surf2light,
                                                                                    light_idx=chunk_light_idx, 
                                                                                    nSample=nSample, 
                                                                                    vis_near=vis_near, 
                                                                                    vis_far=vis_far, 
                                                                                    device=device
                                                                                    ) 


        visibility_chunk = nerv_vis_chunk
        visibility_masked[chunk_idx] = visibility_chunk
        indirect_light_masked[chunk_idx] = indirect_light_chunk

    visibility_compute[invisibile_to_direct_light_mask] = visibility_masked


    # randomly sample some rays uniformly to make supervision of visibility more robust
    uniform_random_sample_mask = (torch.rand(invisibile_to_direct_light_mask.shape, device=device) < 0.4) # [N, 1]
    recompute_visibility_mask = torch.logical_and(uniform_random_sample_mask, ~invisibile_to_direct_light_mask) # [N, 1]

    surface_pts_masked = surface_pts[recompute_visibility_mask] # [masked(N), 3]
    surf2light_masked = surf2light[recompute_visibility_mask] # [masked(N), 3]
    recompute_visibility_masked = torch.zeros((surface_pts_masked.shape[0]), dtype=torch.float32).to(device) # [masked(N), 1]
    chunk_idxs_vis_recompute = torch.split(torch.arange(surface_pts_masked.shape[0]), 20480) # to save memory; TODO: chunk size should be configurable
    for chunk_idx in chunk_idxs_vis_recompute:
        chunk_surface_pts = surface_pts_masked[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = surf2light_masked[chunk_idx]    # [chunk_size, 3]
        nerv_vis, nerfactor_vis = compute_transmittance(tensoIR=tensoIR, 
                                                        surf_pts=chunk_surface_pts, 
                                                        light_in_dir=chunk_surf2light, 
                                                        nSample=nSample, 
                                                        vis_near=vis_near, 
                                                        vis_far=vis_far, 
                                                        device=device
                                                        ) # [N, 1]

        chunk_visibility = nerv_vis

        recompute_visibility_masked[chunk_idx] = chunk_visibility
    

    visibility_compute[recompute_visibility_mask] = recompute_visibility_masked
    visibility_compute = visibility_compute.reshape(-1, 1)  # [N, 1]


    indirect_light[invisibile_to_direct_light_mask] = indirect_light_masked  # [N, 3]
    indirect_light = indirect_light.reshape(-1, 3)  # [N, 3]
    computed_visbility_mask = torch.logical_or(invisibile_to_direct_light_mask, uniform_random_sample_mask) # [N, 1]
    computed_visbility_mask = computed_visbility_mask.reshape(-1, 1)  # [N, 1]

    return visibility_predict, visibility_compute, indirect_light, computed_visbility_mask



@torch.no_grad()
def compute_secondary_shading_effects(
                                        tensoIR, 
                                        surface_pts, 
                                        surf2light, 
                                        light_idx, 
                                        nSample=96, 
                                        vis_near=0.05, 
                                        vis_far=1.5,
                                        chunk_size=15000, 
                                        device='cuda'
                                        ):
    '''compute visibility for each point at each direction without visbility network
    - args:
        - tensoIR: tensoIR model is used to compute the visibility and indirect lighting
        - surface_pts: [N, 3] surface points location
        - surf2light: [N, 3], light incident direction for each surface point, pointing from surface to light
        - light_idx: [N, 1], index of lighitng
        - nSample: number of samples for each ray along incident light direction
    - return:
        - visibility_compute: [N, 1] visibility result by choosing some directions and then computing the density
        - indirect_light: [N, 3] indirect light in the corresponding direction
    '''



    visibility_compute = torch.zeros((surface_pts.shape[0]), dtype=torch.float32).to(device) # [N, 1]
    indirect_light = torch.zeros((surface_pts.shape[0], 3), dtype=torch.float32).to(device) # [N, 1]

    chunk_idxs_vis_compute = torch.split(torch.arange(surface_pts.shape[0]), chunk_size)
    for chunk_idx in chunk_idxs_vis_compute:
        chunk_surface_pts = surface_pts[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = surf2light[chunk_idx]    # [chunk_size, 3]
        chunk_light_idx = light_idx[chunk_idx]      # [chunk_size, 1]
        nerv_vis_chunk, nerfactor_vis_chunk, indirect_light_chunk = compute_radiance(
                                                                                        tensoIR=tensoIR, 
                                                                                        surf_pts=chunk_surface_pts, 
                                                                                        light_in_dir=chunk_surf2light,
                                                                                        light_idx=chunk_light_idx, 
                                                                                        nSample=nSample, 
                                                                                        vis_near=vis_near, 
                                                                                        vis_far=vis_far, 
                                                                                        device=device
                                                                                    ) 


        visibility_chunk = nerv_vis_chunk
        visibility_compute[chunk_idx] = visibility_chunk
        indirect_light[chunk_idx] = indirect_light_chunk


    visibility_compute = visibility_compute.reshape(-1, 1)  # [N, 1]
    indirect_light = indirect_light.reshape(-1, 3)  # [N, 3]


    return visibility_compute, indirect_light



def render_with_BRDF(
        depth_map,
        normal_map,
        albedo_map,
        roughness_map,
        fresnel_map,
        rays,
        tensoIR,
        light_idx,
        sample_method='fixed_envirmap',
        chunk_size=15000,
        device='cuda',
        use_linear2srgb=True,
        args=None
):
    # Relight module
    ## Sample surface points using depth prediction
    surface_z = depth_map  # [bs,]
    rays_o, rays_d = rays[..., :3].to(device), rays[..., 3:].to(device)  # [bs, 3]
    surface_xyz = rays_o + (surface_z).unsqueeze(-1) * rays_d  # [bs, 3]

    ## Get incident light direction
    light_area_weight = tensoIR.light_area_weight.to(device) # [envW * envH, ]

    incident_light_dirs = tensoIR.gen_light_incident_dirs(method=sample_method).to(device)  # [envW * envH, 3]
    surf2l = incident_light_dirs.reshape(1, -1, 3).repeat(surface_xyz.shape[0], 1, 1)  # [bs, envW * envH, 3]
    surf2c = -rays_d  # [bs, 3]
    surf2c = safe_l2_normalize(surf2c, dim=-1)  # [bs, 3]

    ## get visibilty map from visibility network or compute it using density
    cosine = torch.einsum("ijk,ik->ij", surf2l, normal_map)  # surf2l:[bs, envW * envH, 3] * normal_map:[bs, 3] -> cosine:[bs, envW * envH]
    cosine = torch.clamp(cosine, min=0.0)  # [bs, envW * envH]
    cosine_mask = (cosine > 1e-6)  # [bs, envW * envH], mask half of the incident light that is behind the surface
    visibility_compute = torch.zeros((*cosine_mask.shape, 1), device=device)   # [bs, envW * envH, 1]
    indirect_light = torch.zeros((*cosine_mask.shape, 3), device=device)   # [bs, envW * envH, 3]

    visibility_compute[cosine_mask], \
        indirect_light[cosine_mask] = compute_secondary_shading_effects(
                                                        tensoIR=tensoIR,
                                                        surface_pts=surface_xyz.unsqueeze(1).expand(-1, surf2l.shape[1], -1)[cosine_mask],
                                                        surf2light=surf2l[cosine_mask],
                                                        light_idx=light_idx.view(-1, 1, 1).expand((*cosine_mask.shape, 1))[cosine_mask],
                                                        nSample=args.second_nSample,
                                                        vis_near=args.second_near,
                                                        vis_far=args.second_far,
                                                        chunk_size=chunk_size,
                                                        device=device
                                                    )
    visibility_to_use = visibility_compute
    ## Get BRDF specs
    nlights = surf2l.shape[1]
    specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)  # [bs, envW * envH, 3]
    surface_brdf = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular # [bs, envW * envH, 3]


    ## Compute rendering equation
    envir_map_light_rgbs = tensoIR.get_light_rgbs(incident_light_dirs, device=device).to(device) # [light_num, envW * envH, 3]
    direct_light_rgbs = torch.index_select(envir_map_light_rgbs, dim=0, index=light_idx.squeeze(-1)).to(device) # [bs, envW * envH, 3]
    
    light_rgbs = visibility_to_use * direct_light_rgbs + indirect_light # [bs, envW * envH, 3]

    # # no visibility and indirect light
    # light_rgbs = direct_light_rgbs

    # # # no indirect light
    # light_rgbs = visibility_to_use * direct_light_rgbs  # [bs, envW * envH, 3]

    if sample_method == 'stratifed_sample_equal_areas':
        rgb_with_brdf = torch.mean(4 * torch.pi * surface_brdf * light_rgbs * cosine[:, :, None], dim=1)  # [bs, 3]

    else:
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
        rgb_with_brdf = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]
    ### Tonemapping
    rgb_with_brdf = torch.clamp(rgb_with_brdf, min=0.0, max=1.0)  
    ### Colorspace transform
    if use_linear2srgb and rgb_with_brdf.shape[0] > 0:
        rgb_with_brdf = linear2srgb_torch(rgb_with_brdf)
    

    return rgb_with_brdf





def linear2srgb_torch(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    elif isinstance(tensor_0to1, np.ndarray):
        pow_func = np.power
        where_func = np.where
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = _clip_0to1_warn_torch(tensor_0to1)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1 + 1e-6, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def _clip_0to1_warn_torch(tensor_0to1):
    """Enforces [0, 1] on a tensor/array that should be already [0, 1].
    """
    msg = "Some values outside [0, 1], so clipping happened"
    if isinstance(tensor_0to1, torch.Tensor):
        if torch.min(tensor_0to1) < 0 or torch.max(tensor_0to1) > 1:
            logger.debug(msg)
            tensor_0to1 = torch.clamp(
                tensor_0to1, min=0, max=1)
    elif isinstance(tensor_0to1, np.ndarray):
        if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
            logger.debug(msg)
            tensor_0to1 = np.clip(tensor_0to1, 0, 1)
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')
    return tensor_0to1



def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def sph2cart(pts_sph: np.array, convention='lat-lng'):
    # Check input shape
    assert (pts_sph.ndim == 2 and pts_sph.shape[-1] == 3), "Shape of input mush be (n, 3)"
    # Check degree range
    assert (np.abs(pts_sph[:, 1:]) <= 2 * np.pi).all(), "Input degree falls out of [-2pi, 2pi]"

    # Convert to lat-lng convention
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute xyz coord
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)
    pts_cart = np.stack((x, y, z), axis=-1)

    return pts_cart




def read_hdr(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, 'rb') as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb




@torch.no_grad()
def compute_visibility(tensoIR, pts, light_xyz, nSample, vis_near, vis_far, args, device='cuda'):
    '''compute visibility for each point at each direction by calculating the density compostion
    - args:
        - tensoIR: base model
        - pts: [N, 3] surface points
        - light_xyz: [preditected_light_num, 3], locations for each pixel in the environment map
        - nSample: number of samples for each ray along incident light direction
    - return:
        - visibility: [N, preditected_light_num] visibility for each point at each direction
    '''

    surf2light = light_xyz[None, :, :] - pts[:, None, :] # [N, preditected_light_num, 3]
    surf2light = safe_l2_normalize(surf2light, dim=-1)  # [N, preditected_light_num, 3]

    # expand the shape for pts to make it the same as light_xyz
    surface_pts = pts.unsqueeze(1).expand(-1, light_xyz.shape[0], -1)  # [N, preditected_light_num, 3]
    surface_pts = surface_pts.reshape(-1, 3)  # [N*preditected_light_num, 3]
    surf2light = surf2light.reshape(-1, 3)  # [N*preditected_light_num, 3]
    visibility = torch.zeros((surface_pts.shape[0]), dtype=torch.float32).to(device) # [N*preditected_light_num, 1]
    chunk_idxs = torch.split(torch.arange(surface_pts.shape[0]), 81920) # to save memory TODO: chunk size should be configurable
    for chunk_idx in chunk_idxs:
        chunk_surface_pts = surface_pts[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = surf2light[chunk_idx]    # [chunk_size, 3]
        nerv_vis, nerfactor_vis = compute_transmittance(tensoIR=tensoIR, 
                                                        surf_pts=chunk_surface_pts, 
                                                        light_in_dir=chunk_surf2light, 
                                                        nSample=nSample, 
                                                        vis_near=vis_near, 
                                                        vis_far=vis_far, 
                                                        device=device
                                                        ) # [N*preditected_light_num, 1]
        if args.vis_equation == 'nerv':
            chunk_visibility = nerv_vis
        elif args.vis_equation == 'nerfactor':
            chunk_visibility = nerfactor_vis
        visibility[chunk_idx] = chunk_visibility
    visibility = visibility.reshape(-1, light_xyz.shape[0], 1)  # [N, preditected_light_num]
    return visibility


@torch.no_grad()
def compute_transmittance(tensoIR, surf_pts, light_in_dir, nSample=128, vis_near=0.1, vis_far=2, device='cuda'):
    '''same way as in NeRV
    - args:
        - tensoIR: base model
        - surf_pts: [N, 3] surface points locations
        - light_in_dir: [N, 3], normalized light incident direction, pointing from surface to light
        - nSample: number of samples for each ray along incident light direction
        - near: sample begin from this distance
        - far: sample end at this distance
    - return:
        - nerv_vis: [N, preditected_light_num] transmittance for each point at each direction, using the eqaution mentioned in NeRV
        - nerfactor_vis: [N, preditected_light_num] transmittance for each point at each direction, using the eqaution implemented in the code of NeRFactor
    '''

    xyz_sampled, z_vals, ray_valid = sample_ray_equally(tensoIR, 
                                                        surf_pts, 
                                                        light_in_dir, 
                                                        nSample=nSample, 
                                                        vis_near=vis_near, 
                                                        vis_far=vis_far
                                                        )
    dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)


    if tensoIR.alphaMask is not None:
        alphas = tensoIR.alphaMask.sample_alpha(xyz_sampled[ray_valid])
        alpha_mask = alphas > 0
        ray_invalid = ~ray_valid
        ray_invalid[ray_valid] |= (~alpha_mask)
        ray_valid = ~ray_invalid

    # Create empty tensor to store sigma and rgb
    sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

    if ray_valid.any():
        xyz_sampled = tensoIR.normalize_coord(xyz_sampled)
        sigma_feature = tensoIR.compute_densityfeature(xyz_sampled[ray_valid])  # [..., 1] # detach unremoved
        validsigma = tensoIR.feature2density(sigma_feature)
        sigma[ray_valid] = validsigma


    alpha, weight, transimittance = raw2alpha(sigma, dists * tensoIR.distance_scale)

    acc_map = torch.sum(weight, -1) # [N, ]
    nerv_vis = transimittance.squeeze(-1) 
    nerfactor_vis = 1 - acc_map

    return nerv_vis, nerfactor_vis

@torch.no_grad()
def sample_ray_equally(tensoIR, rays_o, rays_d, nSample=-1, vis_near=0.03, vis_far=1.5, device='cuda'):
    '''
    The major differences from the original sample_ray in tensoIR model are:
    1. The near and far are not fixed, but work as input parameters for this function
    2. each ray sample points equally spaced along the ray, without any disturbance
    '''


    t = torch.linspace(0., 1., nSample, device=device) # [nSample,]
    z_vals = (vis_near * (1. - t) + vis_far * t).unsqueeze(0) # [1, nSample]

    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] *  z_vals.view(1, -1, 1) # [N, nSample, 3]
    mask_outbbox = ((tensoIR.aabb[0] > rays_pts) | (rays_pts > tensoIR.aabb[1])).any(dim=-1)

    return rays_pts, z_vals, ~mask_outbbox



@torch.no_grad()
def compute_visibility_and_indirect_light(tensoIR, pts, light_xyz, light_idx, nSample, vis_near, vis_far, args, device='cuda'):
    '''compute visibility and indirect light (represented as radiance field) 
        for each point at each direction by calculating the density compostion
    - args:
        - tensoIR: base model
        - pts: [N, 3] surface points
        - light_xyz: [preditected_light_num, 3], locations for each pixel in the environment map
        - light_idx: [N, 1], index for each pixel in the environment map
        - nSample: number of samples for each ray along incident light direction
    - return:
        - visibility: [N, preditected_light_num, 1] visibility for each point at each direction
        - indirect_light: [N, preditected_light_num, 3] visibility for each point at each direction
    '''

    surf2light = light_xyz[None, :, :] - pts[:, None, :] # [N, preditected_light_num, 3]
    surf2light = safe_l2_normalize(surf2light, dim=-1)  # [N, preditected_light_num, 3]

    light_idx = light_idx.view(-1, 1, 1).expand((-1, surf2light.shape[1], 1)).reshape(-1, 1) # [N*preditected_light_num, 1]

    # expand the shape for pts to make it the same as light_xyz
    surface_pts = pts.unsqueeze(1).expand(-1, light_xyz.shape[0], -1)  # [N, preditected_light_num, 3]
    surface_pts = surface_pts.reshape(-1, 3)  # [N*preditected_light_num, 3]
    surf2light = surf2light.reshape(-1, 3)  # [N*preditected_light_num, 3]

    visibility = torch.zeros((surface_pts.shape[0]), dtype=torch.float32).to(device) # [N*preditected_light_num, 1]
    indirect_light = torch.zeros((surface_pts.shape[0], 3), dtype=torch.float32).to(device) # [N*preditected_light_num, 1]
    chunk_idxs = torch.split(torch.arange(surface_pts.shape[0]), 81920) # to save memory TODO: chunk size should be configurable
    for chunk_idx in chunk_idxs:
        chunk_surface_pts = surface_pts[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = surf2light[chunk_idx]    # [chunk_size, 3]
        chunk_light_idx = light_idx[chunk_idx]     # [chunk_size, 1]
        nerv_vis_chunk, nerfactor_vis_chunk, indirect_light_chunk = compute_radiance(tensoIR=tensoIR, 
                                                                                    surf_pts=chunk_surface_pts, 
                                                                                    light_in_dir=chunk_surf2light,
                                                                                    light_idx=chunk_light_idx, 
                                                                                    nSample=nSample, 
                                                                                    vis_near=vis_near, 
                                                                                    vis_far=vis_far, 
                                                                                    device=device
                                                                                    ) # [N*preditected_light_num, 1]
        if args.vis_equation == 'nerv':
            visibility_chunk = nerv_vis_chunk
        elif args.vis_equation == 'nerfactor':
            visibility_chunk = nerfactor_vis_chunk
        visibility[chunk_idx] = visibility_chunk
        indirect_light[chunk_idx] = indirect_light_chunk
    visibility = visibility.reshape(-1, light_xyz.shape[0], 1)  # [N, preditected_light_num, 1]
    indirect_light = indirect_light.reshape(-1, light_xyz.shape[0], 3)  # [N, preditected_light_num, 3]
    return visibility, indirect_light

@torch.no_grad()
def compute_radiance(tensoIR, surf_pts, light_in_dir, light_idx, nSample=128, vis_near=0.05, vis_far=1.5, device=None):
    '''
    - args:
        - tensoIR: base model
        - surf_pts: [N, 3] surface points locations
        - light_in_dir: [N, 3], normalized light incident direction, pointing from surface to light
        - light_idx: [N, 1], index of light
        - nSample: number of samples for each ray along incident light direction
        - near: sample begin from this distance
        - far: sample end at this distance
    - return:
        - [N, 3] indirect light in the corresponding direction
    '''

    xyz_sampled, z_vals, ray_valid = sample_ray_equally(tensoIR, 
                                                        surf_pts, 
                                                        light_in_dir, 
                                                        nSample=nSample, 
                                                        vis_near=vis_near, 
                                                        vis_far=vis_far
                                                        )
    dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

    light_idx = light_idx.view(-1, 1, 1).expand((*xyz_sampled.shape[:-1], 1)) # (batch_N, n_sammple, 1)
    viewdirs = light_in_dir.view(-1, 1, 3).expand(xyz_sampled.shape) # (batch_N, N_samples, 3)
    if tensoIR.alphaMask is not None:
        alphas = tensoIR.alphaMask.sample_alpha(xyz_sampled[ray_valid])
        alpha_mask = alphas > 0
        ray_invalid = ~ray_valid
        ray_invalid[ray_valid] |= (~alpha_mask)
        ray_valid = ~ray_invalid

    # Create empty tensor to store sigma and rgb
    sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
    indirect_light = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
    if ray_valid.any():
        xyz_sampled = tensoIR.normalize_coord(xyz_sampled)
        sigma_feature = tensoIR.compute_densityfeature(xyz_sampled[ray_valid])  # [..., 1]
        sigma[ray_valid] = tensoIR.feature2density(sigma_feature)


    alpha, weight, transmittance = raw2alpha(sigma, dists * tensoIR.distance_scale)


    app_mask = weight > tensoIR.rayMarch_weight_thres
    if app_mask.any():
        radiance_field_feat = tensoIR.compute_appfeature(xyz_sampled[app_mask], light_idx[app_mask])
        indirect_light[app_mask] = tensoIR.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], radiance_field_feat)

    acc_map = torch.sum(weight, -1) # [N, ]
    nerv_vis = transmittance.squeeze(-1)    # NeRV's way to accumulate visibility
    nerfactor_vis = 1 - acc_map             # nerfactor's way to accumulate visibility

    indirect_light = torch.sum(weight[..., None] * indirect_light, -2)


    return nerv_vis, nerfactor_vis, indirect_light




if __name__ == "__main__":
    pass