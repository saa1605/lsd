


import argparse
from tkinter import TRUE
from src.utils import open_graph


parser = argparse.ArgumentParser(description="SDR task")

parser.add_argument("--freeze_resnet", default=True, action="store_true")
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--annotate_regions", default=False, action="store_true")
parser.add_argument("--annotate_objects", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--data_dir", type=str, default="/data1/saaket/touchdown/data/")
parser.add_argument("--image_dir", type=str, default="/data1/saaket/lsd_data/data/processed/pano_slices")
parser.add_argument("--embedding_dir", type=str, default="/data1/saaket/lsd_data/data/raw/word_embeddings/")
parser.add_argument("--processed_save_path", type=str, default="/data1/saaket/lsd_data/data/processed")
parser.add_argument("--interim_save_path", type=str, default="/data1/saaket/lsd_data/data/interim")
parser.add_argument('--name', type=str, default='run',
                    help='name of the run')
parser.add_argument('--model', type=str, default='lingunet',
                    help='model used')
parser.add_argument('--freeze_image_encoder', default=False, action="store_true")
parser.add_argument('--freeze_text_encoder', default=False, action="store_true")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--bbox_width", type=int, default=5)


parser.add_argument("--checkpoint_dir", type=str, default="/data2/saaket/models/checkpoints/")
parser.add_argument("--predictions_dir", type=str, default="../reports/predictions")
parser.add_argument("--model_save", default=False, action="store_true")
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="/path/to/ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)

parser.add_argument("--box_restriction", type=str, default="top500")
parser.add_argument("--sdr_ds_height", type=int, default=100)
parser.add_argument("--sdr_ds_width", type=int, default=456)
# parser.add_argument("--sdr_ds_width", type=int, default=464)
parser.add_argument("--pano_slices", type=int, default=8)

# CNN
parser.add_argument('--num_conv_layers', type=int, default=1,
                    help='number of conv layers')
parser.add_argument('--conv_dropout', type=float, default=0.0,
                    help='dropout applied to the conv_filters (0 = no dropout)')
parser.add_argument('--deconv_dropout', type=float, default=0.0,
                    help='dropout applied to the deconv_filters (0 = no dropout)')
# RNN
parser.add_argument('--embed_size', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--rnn_hidden_size', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--num_rnn_layers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--bidirectional', type=bool, default=True,
                    help='use bidirectional rnn')
parser.add_argument('--embed_dropout', type=float, default=0.1,
                    help='dropout applied to the embedding layer (0 means no dropout)')


# final linear layers
parser.add_argument('--num_linear_hidden_layers', type=int, default=1,
                    help='number of final linear hidden layers')
parser.add_argument('--linear_hidden_size', type=int, default=128,
                    help='final linear hidden layer size')

# architecture specific arguments
parser.add_argument('--num_rnn2conv_layers', type=int, default=None,
                    help='number of rnn2conv layers')
parser.add_argument('--num_lingunet_layers', type=int, default=None,
                    help='number of LingUNet layers')
parser.add_argument('--num_unet_layers', type=int, default=None,
                    help='number of UNet layers')
parser.add_argument('--num_reslingunet_layers', type=int, default=None,
                    help='number of ResLingUNet layers')

parser.add_argument('--colors',  default=[(240,0,30), (155,50,210), (255,255,25), (0,10,255), (255,170,230), (0,255,0)])
parser.add_argument('--color_names', default=['red', 'purple', 'yellow', 'blue', 'pink', 'green'])
parser.add_argument('--alpha', type=int, default=125)
parser.add_argument('--gaussian_target', type=bool, default=True,
                    help='use Gaussian target')
parser.add_argument('--sample_used', type=float, default=1.0,
                    help='portion of sample used for training')
parser.add_argument('--tuneset_ratio', type=float, default=0.07,
                    help='portion of tune set')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate')
parser.add_argument('--grad_clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--num_epoch', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--summary", default=True, action="store_true", help="tensorboard")
parser.add_argument("--run_name", type=str, default="no_name", help="name of the run")
parser.add_argument("--cuda", type=str, default=0, help="which GPU to use")
parser.add_argument("--max_lr", type=float, default=1e-5, help="initial learning rate")
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument('--clip_version', type=str, default='RN50')

def parse_args():
    args = parser.parse_args()
    return args
