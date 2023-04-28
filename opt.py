import configargparse



def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--hdrdir", type=str, default='./data/llff/fern',
                        help='input HDR directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP', 'ShapeModel'])


    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--save_iters", type=int, default=10000)

    parser.add_argument('--dataset_name', type=str, default='tensoIR_unknown_rotated_lights',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data', 
                        'tensorf_init', 'shapeBuffer', 'tensoIR_unknown_rotated_lights', 'tensoIR_unknown_general_multi_lights',
                        'tensoIR_simple', 'tensoIR_relighting_test', 'tensoIR_material_editing_test', 'tensoIR_simple_dtu'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=2,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=2,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=2,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    


    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)


    parser.add_argument("--test_number", type=int, default=200)

    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')


    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')
    


    parser.add_argument("--rgb_brdf_weight", type=float, default=0.1, help="weight for image loss of physically-based rendering")

    parser.add_argument("--scene_bbox", type=str, action="append")

    parser.add_argument("--second_near", type=float, default=0.05, help='starting point for secondary shading')

    parser.add_argument("--second_far",  type=float, default=1.5, help='ending point for secondary shading')

    parser.add_argument("--second_nSample",  type=int, default=96, help='sampling number along each incoming ray for secondary shading')

    parser.add_argument("--light_sample_train", type=str, default='stratified_sampling')

    parser.add_argument("--light_kind", type=str, default='sg', help='light kind, pixel or spherical gaussian')

    parser.add_argument("--numLgtSGs", type=int, default='128', help='number of spherical gaussian lights')

    parser.add_argument("--light_name", type=str, default="sunset", help="name of the unknown rotated lighting scene")
    
    parser.add_argument("--light_name_list", type=str, action="append")

    parser.add_argument("--light_rotation", type=str, action="append")
    
    parser.add_argument("--acc_thre", type=float, default=0.5, 
                        help="acc_map threshold, less than threshhold will be set to 0")
                        
    parser.add_argument("--geo_buffer_train", action='store_true', default=0)

    parser.add_argument("--geo_buffer_test", action='store_true', default=0)

    parser.add_argument("--geo_buffer_path", type=str, default='.')
   
    parser.add_argument("--echo_every", type=int, default=10, 
                            help="echo loss information every N iterations")
    
    parser.add_argument("--relight_chunk_size", type=int, default=160000, help="chunk size when accumulating the visibility and indirect light")
    
    parser.add_argument("--batch_size_test", type=int, default=4096, help="bath size for test")

    parser.add_argument("--normals_diff_weight", type=float, default=0.0002, help="weight for normals difference loss (control the difference between predicted normals and derived normals)")

    parser.add_argument("--normals_orientation_weight", type=float, default=0.001, help="weight for normals orientation loss, introduced in ref-nerf as Ro loss")

    parser.add_argument("--BRDF_loss_enhance_ratio", type=float, default=1, help="ratio between the final weight and the initial weight for normals diff loss and normals direction loss")

    parser.add_argument("--normals_loss_enhance_ratio", type=float, default=1, help="ratio between the final weight and the initial weight for normals diff loss and normals orientation loss")

    parser.add_argument("--albedo_smoothness_loss_weight", type=float, default=0.0002, help="weight for albedo smooothness loss")

    parser.add_argument("--roughness_smoothness_loss_weight", type=float, default=0.0002, help="weight for roughness smooothness loss")


    parser.add_argument("--normals_kind", type=str, default="derived_plus_predicted", help="ways to get normals",
                        choices=["purely_derived", "purely_predicted", "derived_plus_predicted", "gt_normals", "residue_prediction"])

    # # used to visibility network, deprecated now
    # parser.add_argument("--ckpt_visibility", type=str, help="path to save visibility network checkpoint")

    # parser.add_argument("--vis_model_name", type=str, default='ShapeModel', help="name of visibility network checkpoint")
   
    # parser.add_argument("--train_visibility", action='store_true', help="if train visibility network")

    # parser.add_argument("--visi_lr", default=0.001, type=float, help="learning rate for visibility network" )

    # parser.add_argument("--visibilty_diff_weight", type=float, default=0.0, help="weight for visibility difference loss")



    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()