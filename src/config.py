


import argparse
from src.utils import open_graph


parser = argparse.ArgumentParser(description="LED task")

# What are you doing
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--text_feature_mode", type=str, default='one_utterance')
parser.add_argument("--rpn_mode", type=str, default='conventional')
parser.add_argument('--colors',  default=[(240,0,30), (155,50,210), (255,255,25), (0,10,255), (255,170,230), (0,255,0)])
parser.add_argument('--color_names', default=['red', 'purple', 'yellow', 'blue', 'pink', 'green'])

# Data/Input Paths
parser.add_argument("--data_dir", type=str, default="/data1/saaket/lsd_data/data/raw/way_splits/")
parser.add_argument("--image_dir", type=str, default="/data1/saaket/lsd_data/data/raw/floorplans/")


parser.add_argument("--sdr_data_dir", type=str, default="/data1/saaket/touchdown/data/")
parser.add_argument("--sdr_image_dir", type=str, default="/data1/saaket/jpegs_manhattan_touchdown_2021")

parser.add_argument("--connect_dir", type=str, default="/data1/saaket/lsd_data/data/raw/connectivity/")
parser.add_argument(
    "--geodistance_file", type=str, default="/data1/saaket/lsd_data/data/raw/geodistance_nodes.json"
)
parser.add_argument("--processed_save_path", type=str, default="/data1/saaket/lsd_data/data/processed")
parser.add_argument("--interim_save_path", type=str, default="/data1/saaket/lsd_data/data/interim")
# Output Paths
parser.add_argument("--checkpoint_dir", type=str, default="/data2/saaket/models/checkpoints/")
parser.add_argument("--predictions_dir", type=str, default="../reports/predictions")
parser.add_argument("--model_save", default=False, action="store_true")
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="/path/to/ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)


# FO Layer before lingunet and scaling for the image
parser.add_argument("--freeze_clip", default=True, action="store_true")
parser.add_argument("--ds_percent", type=float, default=0.65)
parser.add_argument("--ds_scale", type=float, default=0.125)
parser.add_argument("--ds_height_crop", type=int, default=54)
parser.add_argument("--ds_width_crop", type=int, default=93)
parser.add_argument("--ds_height", type=int, default=57)
parser.add_argument("--ds_width", type=int, default=97)
# parser.add_argument("--sdr_ds_height", type=int, default=62)
# parser.add_argument("--sdr_ds_width", type=int, default=25)
# parser.add_argument("--panorama_splits", type=int, default=8)
parser.add_argument("--max_floors", type=int, default=5)
# CNN
parser.add_argument("--num_conv_layers", type=int, default=1)
parser.add_argument("--conv_dropout", type=float, default=0.0)
parser.add_argument("--deconv_dropout", type=float, default=0.0)
parser.add_argument("--res_connect", default=True, action="store_true")

# Final linear layers
parser.add_argument("--num_linear_hidden_layers", type=int, default=1)
parser.add_argument("--linear_hidden_size", type=int, default=512)
parser.add_argument("--num_lingunet_layers", type=int, default=3)

# Params
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--summary", default=True, action="store_true", help="tensorboard")
parser.add_argument("--run_name", type=str, default="no_name", help="name of the run")
parser.add_argument("--cuda", type=str, default=0, help="which GPU to use")
parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate")
parser.add_argument("--max_lr", type=float, default=1e-5, help="initial learning rate")
parser.add_argument("--grad_clip", type=float, default=0.5, help="gradient clipping")
parser.add_argument("--num_epoch", type=int, default=20, help="upper epoch limit")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument("--max_regions", type=int, default=30)
parser.add_argument('--clip_version', type=str, default='RN50')

# Get scene graphs
def collect_graphs(args):
    scan_graphs = {}
    scans = [s.strip() for s in open(args.connect_dir + "scans.txt").readlines()]
    for scan_id in scans:
        scan_graphs[scan_id] = open_graph(args.connect_dir, scan_id)
    return scan_graphs


def parse_args():
    args = parser.parse_args()
    args.scan_graphs = collect_graphs(args)
    return args
