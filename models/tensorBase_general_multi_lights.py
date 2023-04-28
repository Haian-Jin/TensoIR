import torch
import torch.nn as nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from models.relight_utils import linear2srgb_torch
from dataLoader.ray_utils import safe_l2_normalize

# from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss

def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts



def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4]) 
    lgtMu = torch.abs(lgtSGs[:, 4:]) 
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy
    
def fibonacci_sphere(samples=1):
    '''
    uniformly distribute points on a sphere
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - z * z)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def render_envmap_sg(lgtSGs, viewdirs):
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
        (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    return rgb
    

def compute_envmap(lgtSGs, H, W, tensorfactor):
    '''
    compute environment map from spherical Gaussian light sources
    '''
         
    rgb = render_envmap_sg(lgtSGs, tensorfactor.fixed_viewdirs)
    envmap = rgb.reshape((H, W, 3))
    return envmap



class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPBRDF_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, outc=1, act_net=nn.Sigmoid()):
        super(MLPBRDF_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        self.outc = outc
        self.act_net = act_net
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, outc)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        spec = self.mlp(mlp_in)
        spec = self.act_net(spec)

        return spec






class MLPBRDF_PEandFeature(torch.nn.Module):
    def __init__(self, inChanel, pospe=6, feape=6, featureC=128, outc=1, act_net=nn.Sigmoid()):
        super(MLPBRDF_PEandFeature, self).__init__()

        self.in_mlpC = 2 * pospe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.pospe = pospe
        self.feape = feape
        self.outc = outc
        self.act_net = act_net
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, outc)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, features):
        indata = [features, pts]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        mlp_in = torch.cat(indata, dim=-1)
        spec = self.mlp(mlp_in)
        spec = self.act_net(spec)

        return spec

class MLPNormal_normal_and_xyz(torch.nn.Module):
    def __init__(self, inChanel, feape=6, featureC=128, outc=1, act_net=nn.Sigmoid()):
        super(MLPNormal_normal_and_xyz, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel + 3 + 3
        self.feape = feape
        self.outc = outc
        self.act_net = act_net
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, outc)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, normal, features):
        indata = [pts, normal, features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        spec = self.mlp(mlp_in)
        spec = self.act_net(spec)

        return spec


class MLPNormal_normal_and_PExyz(torch.nn.Module):
    def __init__(self, inChanel, pospe=6, feape=6, featureC=128, outc=1, act_net=nn.Sigmoid()):
        super(MLPNormal_normal_and_PExyz, self).__init__()

        self.in_mlpC = 2 * pospe * 3 + 2 * feape * inChanel + 3 + inChanel + 3
        self.feape = feape
        self.pospe = pospe
        self.outc = outc
        self.act_net = act_net
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, outc)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, normal, features):
        indata = [pts, normal, features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        mlp_in = torch.cat(indata, dim=-1)
        spec = self.mlp(mlp_in)
        spec = self.act_net(spec)

        return spec

class MLPBRDF_onlyFeature(torch.nn.Module):
    def __init__(self, inChanel, pospe=6, feape=6, featureC=128, outc=1, act_net=nn.Sigmoid()):
        super(MLPBRDF_onlyFeature, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel
        self.pospe = pospe
        self.feape = feape
        self.outc = outc
        self.act_net = act_net
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, outc)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        spec = self.mlp(mlp_in)
        spec = self.act_net(spec)

        return spec


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, 
                 density_n_comp=8, 
                 appearance_n_comp=24,
                 app_dim=27,
                 shadingMode='MLP_PE',  
                 alphaMask=None, 
                 near_far=[2.0, 6.0],
                 density_shift=-10, 
                 alphaMask_thres=0.001, 
                 distance_scale=25, 
                 rayMarch_weight_thres=0.0001,
                 pos_pe=2, view_pe=2, fea_pe=2, 
                 featureC=128, 
                 step_ratio=2.0,
                 fea2denseAct='softplus',
                 normals_kind="purely_predicted",
                 light_rotation=['000', '120', '240'],
                 light_name_list = ["sunset", "snow", "courtyard"],
                 envmap_w=32,
                 envmap_h=16,
                 light_kind='pixel',
                 dataset= None,
                 numLgtSGs=128,
                 fixed_fresnel= 0.04,
                 **kwargs
                 ):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio
        self.shadingMode, self.normals_kind, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, normals_kind, pos_pe, view_pe, fea_pe, featureC
        self.light_num = len(light_name_list)
        self.light_rotation = [int(rotation) for rotation in light_rotation]
        self.light_name_list = light_name_list
        self.envmap_w = envmap_w
        self.envmap_h = envmap_h
        self.dataset = dataset
        self.light_kind = light_kind
        self.numLgtSGs = numLgtSGs
        self.fixed_fresnel = fixed_fresnel
        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]
        self.init_svd_volume(gridSize[0], device)
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)
        self.init_light()

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)

        if self.normals_kind == "purely_predicted" or self.normals_kind == "derived_plus_predicted":
            self.renderModule_normal = MLPBRDF_PEandFeature(self.app_dim, pos_pe, fea_pe, featureC, outc=3, 
                                                            act_net=nn.Tanh()).to(device)

        elif self.normals_kind == "residue_prediction":
            self.renderModule_normal = MLPNormal_normal_and_PExyz(self.app_dim, pos_pe, fea_pe, featureC, outc=3, 
                                                            act_net=nn.Tanh()).to(device)
        # 4 = 3 + 1: albedo + roughness
        self.renderModule_brdf= MLPBRDF_PEandFeature(self.app_dim, pos_pe, fea_pe, featureC, outc=4, 
                                                                act_net=nn.Sigmoid()).to(device)


        print("renderModule_brdf", self.renderModule_brdf)
    def generate_envir_map_dir(self, envmap_h, envmap_w, is_jittor=False):
        lat_step_size = np.pi / envmap_h
        lng_step_size = 2 * np.pi / envmap_w
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

        sin_phi = torch.sin(torch.pi / 2 - phi)  # [envH, envW]
        light_area_weight = 4 * torch.pi * sin_phi / torch.sum(sin_phi)  # [envH, envW]
        assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
        light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]
        if is_jittor:
            phi_jittor, theta_jittor = lat_step_size * (torch.rand_like(phi) - 0.5),  lng_step_size * (torch.rand_like(theta) - 0.5)
            phi, theta = phi + phi_jittor, theta + theta_jittor

        view_dirs = torch.stack([   torch.cos(theta) * torch.cos(phi), 
                                    torch.sin(theta) * torch.cos(phi), 
                                    torch.sin(phi)], dim=-1).view(-1, 3)    # [envH * envW, 3]
        
        return light_area_weight, view_dirs

    def init_light(self):
        self.light_area_weight, self.fixed_viewdirs = self.generate_envir_map_dir(self.envmap_h, self.envmap_w)
        nlights = self.envmap_w * self.envmap_h

        if self.light_kind == 'pixel':
            self._light_rgbs = torch.nn.Parameter(torch.FloatTensor(nlights, 3).uniform_(0, 3).to(torch.float32)) # [envH * envW, 3]
        elif self.light_kind == 'sg':
            self.lgtSGs_list = []
            for i in range(self.light_num):
                lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
                lgtSGs.data[:, -2:] = lgtSGs.data[:, -3:-2].expand((-1, 2))

                # make sure lambda is not too close to zero
                lgtSGs.data[:, 3:4] = 10. + torch.abs(lgtSGs.data[:, 3:4] * 20.)
                # init envmap energy
                energy = compute_energy(lgtSGs.data)
                lgtSGs.data[:, 4:] = torch.abs(lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
                energy = compute_energy(lgtSGs.data)

                # deterministicly initialize lobes
                lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
                lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
                lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)
                self.lgtSGs_list.append(lgtSGs)



    def gen_light_incident_dirs(self, sample_number=-1, method='fixed_envirmap', device='cuda'):

        ''' This function is used to generate light incident directions per iteraration, 
            and this function is used for the light kind of 'sg'
        - args:
            - sample_number: sampled incident light directions, this argumet is not always used
            - method:  
                    'fixed_envirmap': generate light incident directions on the fixed center points of the environment map
                    'uniform_sample': sample incident direction uniformly on the unit sphere, sample number is specified by sample_number
                    'stratified_sampling': random sample incident direction on each grid of envirment map
                    'importance_sample': sample based on light energy
        - return:
            - light_incident_directions: [out_putsample_number, 3]
        '''
        if method == 'fixed_envirmap':
            light_incident_directions = self.fixed_viewdirs
        elif method == 'uniform_sample':
            # uniform sampling 'sample_number' points on a unit sphere
            pass # TODO
        elif method == 'stratified_sampling':
            lat_step_size = np.pi / self.envmap_h
            lng_step_size = 2 * np.pi / self.envmap_w

            phi_begin, theta_begin = torch.meshgrid([
                                        torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, self.envmap_h), 
                                        torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, self.envmap_w)
                                        ], 
                                        indexing='ij')
            phi_jittor, theta_jittor = lat_step_size * (torch.rand_like(phi_begin) - 0.5),  lng_step_size * (torch.rand_like(theta_begin) - 0.5)

            phi, theta = phi_begin + phi_jittor, theta_begin + theta_jittor
            
            light_incident_directions = torch.stack([torch.cos(theta) * torch.cos(phi), 
                                        torch.sin(theta) * torch.cos(phi), 
                                        torch.sin(phi)], dim=-1)    # [H, W, 3]

        elif method == 'stratifed_sample_equal_areas':

            sin_phi_size = 2 / self.envmap_h
            lng_step_size = 2 * np.pi / self.envmap_w
 

            sin_phi_begin, theta_begin = torch.meshgrid([torch.linspace(1 - 0.5 * sin_phi_size, -1 + 0.5 * sin_phi_size, self.envmap_h), 
                                                        torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, self.envmap_w)], indexing='ij')

            sin_phi_jittor, theta_jittor = sin_phi_size * (torch.rand_like(sin_phi_begin) - 0.5),  lng_step_size * (torch.rand_like(theta_begin) - 0.5)

            sin_phi, theta = sin_phi_begin + sin_phi_jittor, theta_begin + theta_jittor

            phi = torch.asin(sin_phi)
            light_incident_directions = torch.stack([torch.cos(theta) * torch.cos(phi), 
                            torch.sin(theta) * torch.cos(phi), 
                            torch.sin(phi)], dim=-1)    # [H, W, 3]

        
        elif method == 'importance_sample':
            _, view_dirs = self.generate_envir_map_dir(128, 256, is_jittor=True)
            envir_map = self.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
            with torch.no_grad():
                envir_map = envir_map.reshape(128, 256, 3)

                # compute the pdf of importance sampling of the environment map
                light_intensity = torch.sum(envir_map, dim=2, keepdim=True) # [H, W, 1]
                env_map_h, env_map_w, _ = light_intensity.shape
                h_interval = 1.0 / env_map_h
                sin_theta = torch.sin(torch.linspace(0 + 0.5 * h_interval, np.pi - 0.5 * h_interval, env_map_h)).to(device) # [H, ]
                pdf = light_intensity * sin_theta.view(-1, 1, 1) # [H, W, 1]
                pdf_to_sample = pdf / torch.sum(pdf)  # [H, W, 1]
                pdf_to_compute = pdf_to_sample * env_map_h * env_map_w / (2 * np.pi * np.pi * sin_theta.view(-1, 1, 1)) 

                light_dir_idx = torch.multinomial(pdf_to_sample.view(-1), sample_number, replacement=True) # [sample_number, ]
                envir_map_dir = view_dirs.view(-1, 3).to(device)

                light_dir = envir_map_dir.gather(0, light_dir_idx.unsqueeze(-1).expand(-1, 3)).view(-1, 3) # [num_samples, 3]
                # sample the light rgbs
                envir_map_rgb = envir_map.view(-1, 3)
                light_rgb = envir_map_rgb.gather(0, light_dir_idx.unsqueeze(-1).expand(-1, 3)).view(-1, 3) # [num_samples, 3]
                envir_map_pdf = pdf_to_compute.view(-1, 1)
                light_pdf = envir_map_pdf.gather(0, light_dir_idx.unsqueeze(-1).expand(-1, 1)).view(-1, 1) # [num_samples, 1]

                return light_dir, light_rgb, light_pdf
            
        return light_incident_directions.reshape(-1, 3) # [output_sample_number, 3]
        

    def get_light_rgbs(self, incident_light_directions=None, device='cuda'):
        '''
        - args:
            - incident_light_directions: [sample_number, 3]
        - return: 
            - light_rgbs: [light_num, sample_number, 3]
        '''
        init_light_directions = incident_light_directions.to(device).reshape(-1, 3) # [sample_number, 3]

        if self.light_kind == 'sg':
            light_rgbs_list = []
            for light_kind_idx in range(self.light_num):
                cur_light_rgbs = render_envmap_sg(self.lgtSGs_list[light_kind_idx].to(device), init_light_directions).reshape(-1, 3) # [sample_number, 3]
                light_rgbs_list.append(cur_light_rgbs)
            light_rgbs = torch.stack(light_rgbs_list, dim=0) # [light_num, sample_number, 3]
        else:
            pass
            # if self.light_kind == 'pixel':
            #     environment_map = torch.nn.functional.softplus(self._light_rgbs, beta=5).reshape(self.envmap_h, self.envmap_w, 3).to(device) # [H, W, 3]
            # elif self.light_kind == 'gt':
            #     environment_map = self.dataset.lights_probes.requires_grad_(False).reshape(self.envmap_h, self.envmap_w, 3).to(device) # [H, W, 3]
            # else:
            #     print("Illegal light kind: {}".format(self.light_kind))
            #     exit(1)
            # environment_map = environment_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
            # phi = torch.arccos(remapped_light_directions[:, 2]).reshape(-1) - 1e-6
            # theta = torch.atan2(remapped_light_directions[:, 1], remapped_light_directions[:, 0]).reshape(-1)
            # # normalize to [-1, 1]
            # query_y = (phi / np.pi) * 2 - 1
            # query_x = - theta / np.pi
            # grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
    
            # light_rgbs = F.grid_sample(environment_map, grid, align_corners=False).squeeze().permute(1, 0).reshape(self.light_num, -1, 3)
        return light_rgbs

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass


    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_densityfeature_with_xyz_grad(self, xyz_sampled):
        pass

    def compute_bothfeature(self, xyz_sampled, light_idx_sampled):
        pass

    def compute_intrinfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled, light_idx_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize': self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,
            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'normals_kind': self.normals_kind,
            'light_num': self.light_num,
            'light_kind':self.light_kind,
            'numLgtSGs':self.numLgtSGs,
            'light_name_list':self.light_name_list,

        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device),
                                           alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox
    def get_mid_and_interval(self, batch_size, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples

        s = torch.linspace(0, 1, N_samples+1).cuda()
        m = (s[1:] + s[:-1]) * 0.5
        m = m[None].repeat(batch_size,1)
        interval = 1 / N_samples
        return m , interval

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, N_samples=256, chunk=10240 * 5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rays.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], mask_filtered
    
    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    @torch.enable_grad()
    def compute_derived_normals(self, xyz_locs):
        xyz_locs.requires_grad_(True)
        sigma_feature = self.compute_densityfeature_with_xyz_grad(xyz_locs)  # [..., 1]  detach() removed in the this function
        sigma = self.feature2density(sigma_feature)
        d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)

        gradients = torch.autograd.grad(
                                    outputs=sigma,
                                    inputs=xyz_locs,
                                    grad_outputs=d_output,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True
                                    )[0]
        derived_normals = -safe_l2_normalize(gradients, dim=-1)
        derived_normals = derived_normals.view(-1, 3)
        return derived_normals

    def compute_relative_smoothness_loss(self, values, values_jittor):

        base = torch.maximum(values, values_jittor).clip(min=1e-6)
        difference = torch.sum(((values - values_jittor) / base)**2, dim=-1, keepdim=True)  # [..., 1]

        return difference




    def forward(self, rays_chunk, light_idx, white_bg=True, is_train=False, ndc_ray=False, is_relight=True, N_samples=-1):
        '''
        - args:
            - rays_chunk: (batch_N, 6), batch_N is the number of rays in a batch
            - light_idx: (batch_N, 1) the index of light in the scene
        '''
        viewdirs = rays_chunk[:, 3:6] # (batch_N, 3)

        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                 N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                              dim=-1)  # dist between 2 consecutive points along a ray
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm  # [1, n_sample]
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                             N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape) # (batch_N, N_samples, 3)
        light_idx = light_idx.view(-1, 1, 1).expand((*xyz_sampled.shape[:-1], 1)) # (batch_N, n_sammple, 1)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        # Create empty tensor to store sigma and rgb
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        # Create empty tensor to store normal, roughness, fresnel
        
        normal = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        albedo = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        roughness = torch.zeros((*xyz_sampled.shape[:-1], 1), device=xyz_sampled.device)

        albedo_smoothness_cost = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)
        roughness_smoothness_cost = torch.zeros((*xyz_sampled.shape[:-1], 1), device=xyz_sampled.device)

        normals_diff = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)
        normals_orientation_loss = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)


        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            radiance_field_feat, intrinsic_feat = self.compute_bothfeature(xyz_sampled[app_mask], light_idx[app_mask])
            
            # RGB
            rgb[app_mask] = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], radiance_field_feat)
            if is_relight: 
                # BRDF
                valid_brdf = self.renderModule_brdf(xyz_sampled[app_mask], intrinsic_feat)
                valid_albedo, valid_roughness = valid_brdf[..., :3], (valid_brdf[..., 3:4] * 0.9 + 0.09)
                albedo[app_mask] = valid_albedo         # [..., 3]
                roughness[app_mask] = valid_roughness   # [..., 1]
                
                xyz_sampled_jittor = xyz_sampled[app_mask] + torch.randn_like(xyz_sampled[app_mask]) * 0.01
                intrinsic_feat_jittor = self.compute_intrinfeature(xyz_sampled_jittor)
                valid_brdf_jittor = self.renderModule_brdf(xyz_sampled_jittor, intrinsic_feat_jittor)
                valid_albedo_jittor, valid_roughness_jittor = valid_brdf_jittor[..., :3], (valid_brdf_jittor[..., 3:4] * 0.9 + 0.09)

                albedo_smoothness_cost[app_mask] = self.compute_relative_smoothness_loss(valid_albedo, valid_albedo_jittor)  # [..., 1]
                roughness_smoothness_cost[app_mask] = self.compute_relative_smoothness_loss(valid_roughness, valid_roughness_jittor)  # [..., 1]

                # Normal
                if self.normals_kind == "purely_predicted":
                    valid_normals = self.renderModule_normal(xyz_sampled[app_mask], intrinsic_feat) 
                
                elif self.normals_kind == "purely_derived":
                    valid_normals = self.compute_derived_normals(xyz_sampled[app_mask])
                elif self.normals_kind == "gt_normals":
                    valid_normals = torch.zeros_like(xyz_sampled[app_mask]) # useless
                elif self.normals_kind == "derived_plus_predicted": 
                    # use the predicted normals and penalize the difference between the predicted normals and derived normas at the same time
                    derived_normals = self.compute_derived_normals(xyz_sampled[app_mask])
                    predicted_normals = self.renderModule_normal(xyz_sampled[app_mask], intrinsic_feat)
                    valid_normals = predicted_normals

                    normals_diff[app_mask] = torch.sum(torch.pow(predicted_normals - derived_normals, 2), dim=-1, keepdim=True)
                    normals_orientation_loss[app_mask] = torch.sum(viewdirs[app_mask] * predicted_normals, dim=-1, keepdim=True).clamp(min=0) 
                    
                elif self.normals_kind == "residue_prediction":    
                    derived_normals = self.compute_derived_normals(xyz_sampled[app_mask])
                    predicted_normals = self.renderModule_normal(xyz_sampled[app_mask], derived_normals, intrinsic_feat)
                    valid_normals = predicted_normals

                    normals_diff[app_mask] = torch.sum(torch.pow(predicted_normals - derived_normals, 2), dim=-1, keepdim=True)
                    normals_orientation_loss[app_mask] = torch.sum(viewdirs[app_mask] * predicted_normals, dim=-1, keepdim=True).clamp(min=0)          

                
                normal[app_mask] = valid_normals

        # alpha composition
        acc_map = torch.sum(weight, -1)
        depth_map = torch.sum(weight * z_vals, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if not is_relight:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
                rgb_map = rgb_map + (1. - acc_map[..., None])
                
            return  rgb_map, depth_map, None, \
                    None, None, None, \
                    acc_map, None, None, None, \
                    None, None
        else:
            normal_map = torch.sum(weight[..., None] * normal, -2)
            normals_diff_map = torch.sum(weight[..., None] * normals_diff, -2)
            normals_orientation_loss_map = torch.sum(weight[..., None] * normals_orientation_loss, -2) 

            albedo_map = torch.sum(weight[..., None] * albedo, -2)  # [..., 3]
            roughness_map = torch.sum(weight[..., None] * roughness, -2)  # [..., ]
            fresnel_map = torch.zeros_like(albedo_map).fill_(self.fixed_fresnel)  # [..., 3]

            albedo_smoothness_cost_map = torch.sum(weight[..., None] * albedo_smoothness_cost, -2)  # [..., 1]
            roughness_smoothness_cost_map = torch.sum(weight[..., None] * roughness_smoothness_cost, -2)  # [..., 1]

            albedo_smoothness_loss = torch.mean(albedo_smoothness_cost_map)
            roughness_smoothness_loss = torch.mean(roughness_smoothness_cost_map)



            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
                rgb_map = rgb_map + (1. - acc_map[..., None])

                normal_map = normal_map + (1 - acc_map[..., None]) * torch.tensor([0.0, 0.0, 1.0],
                                                                                device=normal_map.device)  # Background normal
                # normal_map = normal_map 
                                                                                
                albedo_map = albedo_map + (1 - acc_map[..., None])  # Albedo background should be white
                roughness_map = roughness_map + (1 - acc_map[..., None])
                fresnel_map = fresnel_map + (1 - acc_map[..., None])

            # tone mapping & gamma correction
            rgb_map = rgb_map.clamp(0, 1)
            # Tone mapping to make sure the output of self.renderModule() is in linear space, 
            # and the rgb_map output of this forward() is in sRGB space.
            # By doing this, we can use the output of self.renderModule() to better 
            # represent the indirect illumination, which is implemented in another function.
            if rgb_map.shape[0] > 0:
                rgb_map = linear2srgb_torch(rgb_map)

            albedo_map = albedo_map.clamp(0, 1)
            fresnel_map = fresnel_map.clamp(0, 1)
            roughness_map = roughness_map.clamp(0, 1)
            normal_map = safe_l2_normalize(normal_map, dim=-1)


            acc_mask = acc_map > 0.5 # where there may be intersected surface points

            return  rgb_map, depth_map, normal_map, \
                    albedo_map, roughness_map, fresnel_map, \
                    acc_map, normals_diff_map, normals_orientation_loss_map, acc_mask, \
                    albedo_smoothness_loss, roughness_smoothness_loss
