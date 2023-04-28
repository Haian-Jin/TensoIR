import torch
from opt import config_parser


from renderer import *
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *

args = config_parser()
print(args)

device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)
    



if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)

    export_mesh(args)


