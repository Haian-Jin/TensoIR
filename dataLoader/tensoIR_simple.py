
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataLoader.ray_utils import *
from models.relight_utils import read_hdr
import torch.nn as nn

class TensoIR_Dataset_simple(Dataset):
    def __init__(self,
                 root_dir=None,
                #  hdr_dir=None,
                 split='train',
                 random_test = True,
                 light_names=[],
                 N_vis=-1,
                 downsample=1.0,
                 sub=0,
                 light_rotation=['000', '120', '240'],
                 light_name="sunset",
                 scene_bbox=[[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
                 img_height=900,
                 img_width=1200,
                 near=1.0,
                 far=12.0,
                 test_new_pose=False,
                 **kwargs
                 ):

        assert split in ['train', 'test']
        self.N_vis = N_vis
        self.root_dir = Path(root_dir)
        transforms_file_path = os.path.join(self.root_dir, f'transforms_{split}.json')
        with open(transforms_file_path, 'r') as f:
            self.transforms_json = json.load(f)
        self.split = split
        self.light_rotation = light_rotation
        self.light_names = light_names
        self.light_num = len(self.light_rotation)
        self.split_list = []
        self.chosen_frame_idx = []
        for idx, x in enumerate(self.transforms_json['frames']):
            if self.transforms_json['frames'][x]['light_idx'] < self.light_num:
                self.split_list.append(self.transforms_json['frames'][x]['file_path'])
                self.chosen_frame_idx.append(idx)

        if not random_test:
            # sort split_list and chosen_frame_idx according to the file name
            sorted_idx = np.argsort(self.split_list)
            self.split_list = [self.split_list[i] for i in sorted_idx]
            self.chosen_frame_idx = [self.chosen_frame_idx[i] for i in sorted_idx]

        if sub > 0:
            self.split_list = self.split_list[:sub]

        self.img_wh = (int(int(img_width) / downsample), int(int(img_height) / downsample)) 
        self.white_bg = True
        self.downsample = downsample

        self.transform = self.define_transforms()
        self.light_name = light_name
        self.near_far = [near, far]  

        scene_bbox = [eval(item) for item in scene_bbox]
        self.scene_bbox = torch.tensor(scene_bbox)
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)


        ## Load light data
        # self.hdr_dir = Path(hdr_dir)
        # self.read_lights()


        if split == 'train':
            self.read_stack()
        
        if split == 'test' and test_new_pose:

            def normalize(x: np.ndarray) -> np.ndarray:
                """Normalization helper function."""
                return x / np.linalg.norm(x)

            self.read_stack()
            poses = self.all_poses.numpy()
            centroid = poses[:,:3,3].mean(0)
            radcircle = 1.0 * np.linalg.norm(poses[:,:3,3] - centroid, axis=-1).mean()
            centroid[0] += 0
            centroid[1] += 0
            centroid[2] += 0.5 # dog
            # centroid[2] += 0.4 # bread
            new_up_rad = 30 * np.pi / 180


            target_z = radcircle * np.tan(new_up_rad) * (-1)
            render_poses = []

            for th in np.linspace(0., 2.*np.pi, 150):
                camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), 0])
                    
                up = np.array([0,0,1])
                vec2 = normalize(camorigin)
                vec0 = normalize(np.cross(up, vec2))
                vec1 = normalize(np.cross(vec2, vec0))
                pos = camorigin + centroid
                # rotate to align with new pitch rotation
                lookat = -camorigin
                # rotate the Z axis to the target_z
                lookat[2] = target_z

                lookat = normalize(lookat)
                lookat *= -1
                vec2 = lookat
                vec1 = normalize(np.cross(vec2, vec0))

                p = np.stack([vec0, vec1, vec2, pos], 1)

                render_poses.append(p)

            render_poses = np.stack(render_poses, 0)
            render_poses = np.concatenate([render_poses, np.broadcast_to(poses[0,:3,-1:], render_poses[:,:3,-1:].shape)], -1)
            render_poses = render_poses[...,:4]




            img_wh = self.img_wh
            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            fov = self.transforms_json["camera_angle_x"]
            focal = 0.5 * int(img_wh[0]) / np.tan(0.5 * fov)  # fov -> focal length
            # directions = get_ray_directions_blender(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
            directions = get_ray_directions_blender(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)


            self.test_rays = []
            self.test_w2c = []
            for pose_idx in tqdm(range(render_poses.shape[0])):
                pose = render_poses[pose_idx]
                pose = torch.from_numpy(pose).float()
                c2w = torch.cat([pose, torch.tensor([[0, 0, 0, 1]])], dim=0)
                # c2w = torch.from_numpy(pose)

                # import ipdb; ipdb.set_trace()
                rays_o, rays_d = get_rays(directions, c2w)
                rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]
                self.test_rays.append(rays)
                w2c = torch.inverse(c2w)
                self.test_w2c.append(w2c)
            self.test_rays = torch.stack(self.test_rays, dim=0)
            self.test_w2c = torch.stack(self.test_w2c, dim=0)


            del self.all_rays, self.all_rgbs, self.all_light_idx, self.all_masks, self.all_poses


    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms

    def read_lights(self):
        """
        Read hdr file from local path
        """
        self.lights_probes = None

        hdr_path = self.hdr_dir / f'{self.light_name}.hdr'
        if os.path.exists(hdr_path):
            light_rgb = read_hdr(hdr_path)
            self.envir_map_h, self.envir_map_w = light_rgb.shape[:2]
            light_rgb = light_rgb.reshape(-1, 3)
            light_rgb = torch.from_numpy(light_rgb).float()
            self.lights_probes = light_rgb


    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def read_stack(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_light_idx = []
        self.all_masks = []
        self.all_poses = []
        for idx in tqdm(range(self.__len__())):
            item = self.__getitem__(idx)
            rays = item['rays']
            rgbs = item['rgbs']
            light_idx = item['light_idx']
            self.all_rays += [rays]
            self.all_rgbs += [rgbs.squeeze(0)]
            self.all_light_idx += [light_idx.squeeze(0)]
            self.all_masks += [item['rgbs_mask'].squeeze(0)]
            self.all_poses += [item['c2w'].squeeze(0)]

        self.all_rays = torch.cat(self.all_rays, dim=0)  # [N*H*W, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # [N*H*W, 3]
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)  # [N*H*W, 1]
        self.all_masks = torch.cat(self.all_masks, dim=0)  # [N*H*W, 1]
        self.all_poses = torch.stack(self.all_poses, dim=0)  # [N, 4, 4]
    
    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        item_path = self.split_list[idx]
        if item_path.startswith('./'):
            item_path = item_path[2:]
        frame_idx = self.chosen_frame_idx[idx]
        img_wh = self.img_wh
        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        fov = self.transforms_json["camera_angle_x"]
        focal = 0.5 * int(img_wh[0]) / np.tan(0.5 * fov)  # fov -> focal length
        directions = get_ray_directions_blender(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        cam_trans = self.transforms_json['frames'][str(frame_idx)]["transform_matrix"]
        cam_trans = np.array(cam_trans).reshape(4, 4)
        c2w = torch.FloatTensor(cam_trans)  # [4, 4]
        w2c = torch.linalg.inv(c2w)  # [4, 4]


        light_idx = self.transforms_json['frames'][str(frame_idx)]["light_idx"]
        # light_idx = 0
        light_idx = torch.tensor(light_idx, dtype=torch.int).repeat((img_wh[0] * img_wh[1], 1)) # [H*W, 1]


        img_path = os.path.join(self.root_dir, item_path)
        img = Image.open(img_path)
        img = self.transform(img)  # [4, H, W]
        img = img.view(4, -1).permute(1, 0)  # [H*W, 4]
        ## Blend A to RGB
        img_rgbs = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # [H*W, 3]
        ## Obtain background mask, bg = False
        img_mask = ~(img[:, -1:] == 0)
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]


        item = {
            'img_wh': img_wh,  # (int, int)
            'light_idx': light_idx.view(1, -1, 1),  # [1, H*W, 1]
            'rgbs': img_rgbs.view(1, -1, 3),  # [1, H*W, 3]
            'rgbs_mask': img_mask,  # [H*W, 1]
            'rays': rays,  # [H*W, 6]
            'c2w': c2w,  # [4, 4]
            'w2c': w2c  # [4, 4]
        }
        return item

if __name__ == "__main__":
    from opt import config_parser

    args = config_parser()

    dataset = TensoIR_Dataset_simple(
        root_dir='/home/haian/Dataset/real_captured/dog_all_colmap/images',
        hdr_dir='/home/haian/Dataset/light_probes/low_res_envmaps_rotated/',
        split='train',
        random_test=False,
        downsample=1.0,
        light_rotation=['000']
    )
