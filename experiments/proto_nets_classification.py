"""
Resume the protonets model and then train a simple linear classifier
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet, Fabric
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateClassifier, prepare_classifier_task
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
pretrained_parser = parser.add_mutually_exclusive_group(required=False)
pretrained_parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()

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

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_size=64,
    shuffle=True,
    num_workers=1,
    pin_memory=True
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_size=64,
    shuffle=True,
    num_workers=1,
    pin_memory=True
)
test_data = dataset_class('test')
test_taskloader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)


#########
# Model #
#########
if not args.test:
    model = get_few_shot_encoder(num_input_channels)
    pretrained_filename = f'{PATH}/models/proto_nets/{param_str}.pth'
    checkpoints = torch.load(pretrained_filename)
    model.load_state_dict(checkpoints)
    # remove the last Flatten layer
    model = model[:-1]
    for param in model.parameters():
        param.requires_grad = False
    model = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 8)
        # nn.Linear(64*3*3, 500),
        # nn.Dropout(0.75),
        # nn.Linear(500, 8)
    )
    model.to(device, dtype=torch.double)
else:
    model = get_few_shot_encoder(num_input_channels)
    # remove the last Flatten layer
    model = model[:-1]
    model = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 8)
        # nn.Linear(64*3*3, 500),
        # nn.Dropout(0.75),
        # nn.Linear(500, 8)
    )
    for param in model.parameters():
        param.requires_grad = False
    pretrained_filename = f'{PATH}/models/proto_nets/{param_str}_classifier.pth'
    checkpoints = torch.load(pretrained_filename)
    model.load_state_dict(checkpoints)
    # model = nn.Sequential(
    #         *list(model.children())[:-2],
    #         *list(model.children())[-1:])
    model.to(device, dtype=torch.double)


############
# Training #
############
if not args.test:
    print(f'Training Prototypical classifier on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()


    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr

    callbacks = [
        EvaluateClassifier(
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_classifier_task(),
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}_classifier.pth',
            monitor=f'val_acc',
            save_best_only=True
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}_classifier.csv'),
    ]

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_classifier_task(),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
    )
else:
    optimiser = Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    test(
        model,
        optimiser,
        loss_fn,
        dataloader=test_taskloader,
        prepare_batch=prepare_classifier_task(),
    )
