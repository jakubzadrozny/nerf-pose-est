import os
import torch
import numpy as np
import imageio
import warnings
from scipy.interpolate import CubicSpline
import tqdm

from src.pixelnerf.src.render import NeRFRenderer
from src.pixelnerf.src.model import make_model
from src.pixelnerf.src import util
from src.datasets.bop import BOPDataset
from src.datasets.objects import ObjectsDataset


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="Output video resolution"
    )
    parser.add_argument(
        "--focal",
        type=int,
        default=100,
        help="Focal length used to render output frames"
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

print(device)

scene_ds = BOPDataset("data/lm", split="train_pbr")
dset = ObjectsDataset(scene_ds)

data = dset[args.subset]
images = data["images"]  # (NV, 3, H, W)
poses = data["poses"]  # (NV, 4, 4)
focal = data["focal"]
c = data.get("c")

NV, _, H, W = images.shape

if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".format(
                args.scale, H, W
            )
        )
    H, W = Ht, Wt

net = make_model(conf["model"]).to(device=device)
net.load_weights(args)

renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
).to(device=device)
renderer.n_coarse = 64
renderer.n_fine = 128

render_par = renderer.bind_parallel(
    net, args.gpu_id, simple_output=True).eval()

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

print("Generating rays")

if args.radius == 0.0:
    radius = (z_near + z_far) * 0.5
    print("> Using default camera radius", radius)
else:
    radius = args.radius

print("Radius: ", radius)

# Use 360 pose sequence from NeRF
render_poses = torch.stack([
    util.pose_spherical(angle, args.elevation, radius)
    for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
], dim=0)  # (NV, 4, 4)

out_focal = torch.tensor(args.focal)

render_rays = util.gen_rays(
    render_poses,
    args.res,
    args.res,
    out_focal * args.scale,
    z_near,
    z_far,
).to(device=device)
# (NV, H, W, 8)

source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1
assert not (source >= NV).any()

with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    net.encode(
        images[src_view].unsqueeze(0).to(device=device),
        poses[src_view].unsqueeze(0).to(device=device),
        focal[src_view].unsqueeze(0).to(device=device),
        c=c[src_view].unsqueeze(0).to(device=device),
    )

    print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    for rays in tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
    ):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    # rgb_fine (V*H*W, 3)

    frames = rgb_fine.view(-1, args.res, args.res, 3)

print("Writing video")
vid_name = "{:04}".format(args.subset)
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name,
                        "video" + vid_name + ".mp4")
viewimg_path = os.path.join(
    args.visual_path, args.name, "video" + vid_name + "_view.jpg"
)
imageio.mimwrite(
    vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
)

img_np = (data["images"][src_view].permute(0, 2, 3, 1) * 0.5 + 0.5).numpy()
img_np = (img_np * 255).astype(np.uint8)
img_np = np.hstack((*img_np,))
imageio.imwrite(viewimg_path, img_np)

print("Wrote to", vid_path, "view:", viewimg_path)
