"""
Resume the protonets model and then train a simple linear classifier
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet, Fabric
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateClassifier, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit, test
from few_shot.callbacks import *
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Omniglot',
                    help='which dataset to use (Omniglot | miniImagenet | Fabric')
parser.add_argument('--distance', type=str, default='l2',
                    help='which distance metric to use. (l2 | cosine)')
parser.add_argument('--n-train', type=int, default=1,
                    help='support samples per class for training tasks')
parser.add_argument('--n-test', type=int, default=1,
                    help='support samples per class for validation tasks')
parser.add_argument('--k-train', type=int, default=60,
                    help='number of classes in training tasks')
parser.add_argument('--k-test', type=int, default=5,
                    help='number of classes in validation tasks')
parser.add_argument('--q-train', type=int, default=5,
                    help='query samples per class for training tasks')
parser.add_argument('--q-test', type=int, default=1,
                    help='query samples per class for validation tasks')
args = parser.parse_args()

evaluation_episodes = 1000
test_episodes = 1000
# Arbitrary number of batches of n-shot tasks to generate in one epoch
episodes_per_epoch = 1000

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
elif args.dataset == 'Fabric':
    n_epochs = 200
    dataset_class = Fabric
    num_input_channels = 1
    drop_lr_every = 150
else:
    raise(ValueError, 'Unsupported dataset')

# param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            # f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'
param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv=2_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, evaluation_episodes, args.n_test, args.k_test, args.q_test),
    num_workers=4
)
test_data = dataset_class('test')
test_taskloader = DataLoader(
    test_data,
    batch_sampler=NShotTaskSampler(test_data, test_episodes, args.n_test, args.k_test, args.q_test),
    num_workers=4
)


#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
pretrained_filename = f'{PATH}/models/proto_nets/{param_str}.pth'
checkpoints = torch.load(pretrained_filename)
model.load_state_dict(checkpoints)
for param in model.parameters():
    param.requires_grad = False
model.to(device, dtype=torch.double)


############
# Training #
############
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()

# test(
    # model,
    # optimiser,
    # loss_fn,
    # # dataloader=evaluation_taskloader,
    # dataloader=test_taskloader,
    # prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
    # eval_fn=proto_net_episode,
    # eval_fn_kwargs={'n_shot': args.n_test, 'k_way': args.k_test, 'q_queries': args.q_test, 'train': False,
                    # 'distance': args.distance},
# )

test(
    model,
    optimiser,
    loss_fn,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    eval_fn=proto_net_episode,
    eval_fn_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': False,
                    'distance': args.distance},
)
