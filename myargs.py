import argparse

parser = argparse.ArgumentParser()




######################## Model parameters ########################

parser.add_argument('--classes', default=10, type=int,
                    help='# of classes')

parser.add_argument('--lr', default=2e-3, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0000, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.5, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help=', beta2 for adam')
parser.add_argument('--large', default=False, type=bool,
                    help=', use large network')

parser.add_argument('--num_epoch', default=5000, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=16, type=int,
                    help='input batch size')
parser.add_argument('--batch_size_eval', default=128, type=int,
                    help='input batch size at eval time')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--gpu_ids', default='0',
                    help='which gpus to use in train/eval')

parser.add_argument('--radius', type=float, default=3.5,
                    help="Perturbation 2-norm ball radius")
parser.add_argument('--gaussian_noise', type=float, default=1.0,
                    help="noise for feature extractor")
parser.add_argument('--n_power', type=int, default=1,
                    help="gradient iterations")


######################## Model paths ########################

parser.add_argument('--train_model_path',
                    default='data/models/model_resnet18_84.pt')
parser.add_argument('--eval_model_path',
                    default='data/models/model_resnet18_3.pt')
parser.add_argument('--model_save_path',
                    default='data/models')


args = parser.parse_args()
