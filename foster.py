"""FedAvg"""
import os, argparse, copy, time
import numpy as np
import wandb
import torch
from torch import nn, optim
# federated
from learning import train, test
# utils
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, \
    MultiStepLR, LocalMaskCrossEntropyLoss, str2bool
from utils.config import CHECKPOINT_ROOT
import torchvision.transforms as trn
# NOTE import desired federation
from core import _Federation as Federation
from core import AdversaryCreator
#models
from models.allconv import AllConvNet
from models.wrn_virtual import WideResNet, linear_classifier, WideResNet_Tin, WideResNet_stl, WideResNet_Domain
from VOS_virtual import VOS_train, VOS_train2, VOS_train_prox, inversion_train, topk_inversion_train, topk_inversion_train_prox, visualization, visualization2, visualization_external, get_weights
from VOS_evaluate import VOS_evaluate
from torch.utils.data import Dataset
from oodgen import CentralGen


if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utilsood.tinyimages_80mn_loader import TinyImages

class SimpleDataSet(Dataset):
    """ load synthetic time series data"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
        )

def render_run_name(args, exp_folder):
    """Return a unique run_name from given args."""
    if args.model == 'default':
        args.model = {'Digits': 'digit', 'Cifar10': 'preresnet18', 'DomainNet': 'alex'}[args.data]
    run_name = f'{args.model}'
    if args.width_scale != 1.: run_name += f'x{args.width_scale}'
    run_name += Federation.render_run_name(args)
    # log non-default args
    if args.seed != 1: run_name += f'__seed_{args.seed}'
    # opt
    if args.lr_sch != 'none': run_name += f'__lrs_{args.lr_sch}'
    if args.opt != 'sgd': run_name += f'__opt_{args.opt}'
    if args.batch != 32: run_name += f'__batch_{args.batch}'
    if args.wk_iters != 1: run_name += f'__wk_iters_{args.wk_iters}'
    # slimmable
    if args.no_track_stat: run_name += f"__nts"
    if args.no_mask_loss: run_name += f'__nml'
    # adv train
    if args.adv_lmbd > 0:
        run_name += f'__at{args.adv_lmbd}'
    run_name += f'__at{args.loss_weight}'
    run_name += f'__ex{args.use_external}'
    if args.select_generator != None:
        run_name += f'__ex{args.select_generator}'
    if args.method != 'OE':
        run_name += f'__m{args.method}'
    args.save_path = os.path.join(CHECKPOINT_ROOT, exp_folder)
    if args.score != 'OE':
        run_name += f'__score{args.method}'
    if args.sample_number != 1000:
        run_name += f'__sample{args.sample_number}'
    if args.soft != 0:
        run_name += f'__{args.soft}'
    if args.fl != 'fedavg':
        run_name += f'__m{args.fl}'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_FILE = os.path.join(args.save_path, run_name)
    return run_name, SAVE_FILE


def get_model_fh(data, model, num_classes=10):
    if data == 'Digits':
        if model in ['digit']:
            from nets.models import DigitModel
            ModelClass = DigitModel
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data in ['DomainNet', 'ImageNet']:
        if model in ['alex']:
            from nets.models import AlexNet
            ModelClass = AlexNet
        elif model == 'wrn':
            ModelClass = WideResNet_Domain
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'Cifar10' or data == 'Cifar100':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'tin':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet_Tin
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'stl':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet_stl
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError(f"Unknown dataset: {data}")
    return ModelClass


def fed_test(fed, running_model, val_loaders, verbose, adversary=None):
    mark = 's' if adversary is None else 'r'
    val_acc_list = [None for _ in range(fed.client_num)]
    val_loss_mt = AverageMeter()
    for client_idx in range(fed.client_num):
        fed.download(running_model, client_idx)
        # Test
        val_loss, val_acc = test(running_model, val_loaders[client_idx], loss_fun, device,
                                 adversary=adversary)

        # Log
        val_loss_mt.append(val_loss)
        val_acc_list[client_idx] = val_acc
        if verbose > 0:
            print(' {:<19s} Val {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                'User-'+fed.clients[client_idx], mark.upper(), val_loss, mark.upper(), val_acc))
        wandb.log({
            f"{fed.clients[client_idx]} val_{mark}-acc": val_acc,
        }, commit=False)
    return val_acc_list, val_loss_mt.avg

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    # basic problem setting
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data', type=str, default='Digits', help='data name')
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--width_scale', type=float, default=1., help='model width scale')
    parser.add_argument('--no_track_stat', action='store_true', help='disable BN tracking')
    parser.add_argument('--no_mask_loss', action='store_true', help='disable masked loss for class'
                                                                    ' niid')
    parser.add_argument('--fl', choices=['fedavg', 'fedprox'], default='fedavg')
    # control
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--verbose', type=int, default=0, help='verbose level: 0 or 1')
    # federated
    Federation.add_argument(parser)
    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_sch', type=str, default='multi_step', help='learning rate schedule')
    parser.add_argument('--opt', type=str.lower, default='sgd', help='optimizer')
    parser.add_argument('--iters', type=int, default=300, help='#iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1, help='#epochs in local train')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # adversarial train
    parser.add_argument('--adv_lmbd', type=float, default=0.,
                        help='adv coefficient in [0,1]; default 0 for standard training.')
    parser.add_argument('--test_noise', choices=['none', 'LinfPGD'], default='none')
    # energy reg
    parser.add_argument('--start_iter', type=int, default=1000)
    parser.add_argument('--sample_number', type=int, default=1000)
    parser.add_argument('--select', type=int, default=1)
    parser.add_argument('--select_generator', type=int, default=None)
    parser.add_argument('--sample_from', type=int, default=10000)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Setup for OOD evaluation
    parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Evaluate performance on validation distributions.')
    parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
    parser.add_argument('--method_name', '-m', type=str, default='cifar10_wrn_baseline_0.1_50_40_1_10000_0.08',
                        help='Method name.')
    # EG and benchmark details
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
    parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
    parser.add_argument('--model_name', default='res', type=str)
    parser.add_argument('--use_external', type=str, default='None', help='None|class|dataset|gen_inverse')
    parser.add_argument('--oe_batch_size', type=int, default=1000, help='ood Batch size.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--m_in', type=float, default=-25.,
                        help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7.,
                        help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--score', type=str, default='energy', help='OE|energy|energy_VOS')
    parser.add_argument('--method', type=str, default='energy', help='OE|energy|crossentropy')
    parser.add_argument('--evaluation_score', type=str, default='energy', help='energy|msp|odin')
    parser.add_argument('--soft', type=float, default=0, help='If >0, use soft label for generator')
    parser.add_argument('--visualization', type=bool, default=False, help='If True, visualize')


    args = parser.parse_args()

    set_seed(args.seed)

    # set experiment files, wandb
    exp_folder = os.path.basename(os.path.splitext(__file__)[0]) + f'_{args.data}'
    run_name, SAVE_FILE = render_run_name(args, exp_folder)
    wandb.init(group=run_name[:120], project=exp_folder,
               mode='offline' if args.no_log else 'online',
               config={**vars(args), 'save_file': SAVE_FILE})
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    # /////////////////////////////////
    # ///// Fed Dataset and Model /////
    # /////////////////////////////////
    fed = Federation(args.data, args)
    # Data
    train_loaders, val_loaders, test_loaders = fed.get_data()
    mean_batch_iters = int(np.mean([len(tl) for tl in train_loaders]))
    print(f"  mean_batch_iters: {mean_batch_iters}")

    # Model
    ModelClass = get_model_fh(args.data, args.model)
    if args.model == 'wrn' or args.model == 'allconv':
        running_model = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        global_model = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate,
                                   track_running_stats=not args.no_track_stat).to(device)
    else:
        running_model = ModelClass(
            track_running_stats=not args.no_track_stat, num_classes=fed.num_classes,
            width_scale=args.width_scale,
        ).to(device)
        global_model = ModelClass(
            track_running_stats=not args.no_track_stat, num_classes=fed.num_classes,
            width_scale=args.width_scale,
        ).to(device)
    if args.model == 'wrn':
        user_classifier = linear_classifier(fed.num_classes, args.widen_factor).to(device)
    elif args.model == 'preresnet18':
        user_classifier = linear_classifier(fed.num_classes, model=args.model).to(device)
    elif args.model == 'alex':
        user_classifier = copy.deepcopy(running_model.get_fc())
    if args.use_external == 'dataset':
        # mean and standard deviation of channels of tinnyimage
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        ood_data = TinyImages(transform=trn.Compose(
            [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
             trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

        train_loader_out = torch.utils.data.DataLoader(
            ood_data,
            batch_size=args.oe_batch_size, shuffle=False,
            num_workers=args.prefetch, pin_memory=True)
        train_loader_out.dataset.offset = 1000

    user_class = {}
    userlogistic = {}
    weight_energy = {}
    privacy_engine = {}
    Running_model = {}
    # prepare external class set
    for client_idx in range(fed.client_num):
        ## get class for this client
        len_train = len(train_loaders[client_idx])
        label = []
        count = 0
        for batch_id, (data, y) in enumerate(train_loaders[client_idx]):
            count += y.shape[0]
            for j in range(y.shape[0]):
                if y[j] not in label:
                    label.append(y[j])
        user_class[client_idx] = label
        userlogistic[client_idx] = torch.nn.Linear(1, 2)
        userlogistic[client_idx] = userlogistic[client_idx].cuda()
        weight_energy[client_idx] = torch.nn.Linear(args.pu_nclass, 1).cuda()
        torch.nn.init.uniform_(weight_energy[client_idx].weight)
        if args.model == 'wrn' or args.model == 'allconv':
            Running_model[client_idx] = ModelClass(args.layers, fed.num_classes, args.widen_factor,
                                                   dropRate=args.droprate,
                                                   track_running_stats=not args.no_track_stat).to(device)
        else:
            Running_model[client_idx] = ModelClass(
                track_running_stats=not args.no_track_stat, num_classes=fed.num_classes,
                width_scale=args.width_scale,
            ).to(device)


    # adversary
    if args.adv_lmbd > 0. or args.test:
        make_adv = AdversaryCreator(args.test_noise if args.test else 'LinfPGD')
        adversary = make_adv(running_model)
    else:
        adversary = None

    # Loss
    if args.pu_nclass > 0 and not args.no_mask_loss:  # niid
        loss_fun = LocalMaskCrossEntropyLoss(fed.num_classes)
    else:
        loss_fun = nn.CrossEntropyLoss()

    # Use running model to init a fed aggregator
    fed.make_aggregator(running_model, local_fc=args.local_fc)

    # /////////////////
    # //// Resume /////
    # /////////////////
    # log the best for each model on all datasets
    best_epoch = 0
    best_acc = [0. for j in range(fed.client_num)]
    train_elapsed = [[] for _ in range(fed.client_num)]
    start_epoch = 0
    if args.resume or args.test:
        if os.path.exists(SAVE_FILE):
            print(f'Loading chkpt from {SAVE_FILE}')
            checkpoint = torch.load(SAVE_FILE)
            best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
            train_elapsed = checkpoint['train_elapsed']
            start_epoch = int(checkpoint['a_iter']) + 1
            fed.model_accum.load_state_dict(checkpoint['server_model'])

            print('Resume training from epoch {} with best acc:'.format(start_epoch))
            for client_idx, acc in enumerate(best_acc):
                print(' Best user-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                    fed.clients[client_idx], best_epoch, acc))
        else:
            if args.test:
                raise FileNotFoundError(f"Not found checkpoint at {SAVE_FILE}")
            else:
                print(f"Not found checkpoint at {SAVE_FILE}\n **Continue without resume.**")

    # ///////////////
    # //// Test /////
    # ///////////////
    if args.test:
        wandb.summary[f'best_epoch'] = best_epoch

        # Set up model with specified width
        print(f"  Test model: {args.model}x{args.width_scale}"
              + ('' if args.test_noise == 'none' else f'with {args.test_noise} noise'))

        # Test on clients
        if args.data == 'Cifar10':
            dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
        else:
            dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN"]
        auroc_mt, aupr_mt, fpr_mt, test_acc_mt = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        auroc_detail, aupr_detail, fpr_detail = {}, {}, {}
        for i in range(len(dataset_name)):
            auroc_detail[dataset_name[i]], aupr_detail[dataset_name[i]], fpr_detail[dataset_name[i]] = AverageMeter(), AverageMeter(), AverageMeter()
        if args.data == 'tin':
            test_loaders = val_loaders
        for test_idx, test_loader in enumerate(test_loaders):
            fed.download(running_model, test_idx)
            _, test_acc = test(running_model, test_loader, loss_fun, device,
                               adversary=adversary)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[test_idx], test_acc))
            auroc, aupr, fpr, auroc_list, aupr_list, fpr_list = VOS_evaluate(args, args.out_as_pos, args.num_to_avg, args.use_xent, args.method_name, args.evaluation_score, args.test_batch, args.T, args.noise, running_model,
                         test_loader, train_loaders[test_idx], user_class[test_idx], data_name=args.data, m_name=args.use_external, client_id=test_idx)

            wandb.summary[f'{fed.clients[test_idx]} test acc'] = test_acc
            test_acc_mt.append(test_acc)
            wandb.summary[f'{fed.clients[test_idx]} auroc'] = auroc
            wandb.summary[f'{fed.clients[test_idx]} aupr'] = aupr
            wandb.summary[f'{fed.clients[test_idx]} fpr'] = fpr
            auroc_mt.append(auroc)
            aupr_mt.append(aupr)
            fpr_mt.append(fpr)
            for i in range(len(dataset_name)):
                auroc_detail[dataset_name[i]].append(auroc_list[i])
                aupr_detail[dataset_name[i]].append(aupr_list[i])
                fpr_detail[dataset_name[i]].append(fpr_list[i])

        # Profile model FLOPs, sizes (#param)
        from nets.profile_func import profile_model

        flops, params = profile_model(running_model, device=device)
        wandb.summary['GFLOPs'] = flops / 1e9
        wandb.summary['model size (MB)'] = params / 1e6
        print('GFLOPS: %.4f, model size: %.4fMB' % (flops / 1e9, params / 1e6))

        print(f"\n Average Test auroc: {auroc_mt.avg}")
        print(f"\n Average Test aupr: {aupr_mt.avg}")
        print(f"\n Average Test fpr: {fpr_mt.avg}")
        print(f"\n Average Test Acc: {test_acc_mt.avg}")
        wandb.summary[f'avg test acc'] = test_acc_mt.avg
        wandb.summary[f'avg test auroc'] = auroc_mt.avg
        wandb.summary[f'avg test aupr'] = aupr_mt.avg
        wandb.summary[f'avg test fpr'] = fpr_mt.avg
        print("Show detail:")
        for i in range(len(dataset_name)):
            print("{} detection".format(dataset_name[i]))
            print("auroc: {}, aupr: {}, fpr: {}".format(auroc_detail[dataset_name[i]].avg, aupr_detail[dataset_name[i]].avg, fpr_detail[dataset_name[i]].avg))

        wandb.finish()

        exit(0)

    if args.use_external == 'class':
    #if 1:
        if os.path.exists("external_loader2.pth"):
            print("load external classifier2!")
            external_loader = torch.load("external_loader2.pth")
        else:
            external_loader = {}
            # prepare external class set
            for client_idx in range(fed.client_num):
                start = False
                for c in range(fed.client_num):
                    if c != client_idx:
                        for batch_id, (data, y) in enumerate(train_loaders[c]):
                            select_id = [idx for idx in range(y.shape[0]) if y[idx] not in user_class[client_idx]]
                            if start == False:
                                external_x = data[select_id]
                                external_y = y[select_id]
                                start = True
                            else:
                                external_x = torch.cat((external_x, data[select_id]), dim=0)
                                external_y = torch.cat((external_y, y[select_id]), dim=0)
                external_set = SimpleDataSet(external_x, external_y)
                print("external class size", len(external_set))
                print("external label", torch.unique(external_y))
                #externalclass_loader = torch.utils.data.DataLoader(external_set, batch_size=args.oe_batch_size,
                #                                                   shuffle=True,
                #                                                   num_workers=args.prefetch, pin_memory=True)
                externalclass_loader = torch.utils.data.DataLoader(external_set, batch_size=args.batch,
                                                                   shuffle=True,
                                                                   num_workers=args.prefetch, pin_memory=True)
                external_loader[client_idx] = externalclass_loader
            torch.save(external_loader, "external_loader2.pth")


    if args.use_external == 'gen_inverse':
        max_iter = mean_batch_iters * args.wk_iters
        Central_gen = CentralGen(args, max_iter, fed.num_classes, model=args.model)

    # ////////////////
    # //// Train /////
    # ////////////////
    # LR scheduler
    if args.lr_sch == 'cos':
        lr_sch = CosineAnnealingLR(args.iters, eta_max=args.lr, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step':
        lr_sch = MultiStepLR(args.lr, milestones=[150, 250], gamma=0.1, last_epoch=start_epoch)
    else:
        assert args.lr_sch == 'none', f'Invalid lr_sch: {args.lr_sch}'
        lr_sch = None
    total_iter = torch.zeros((fed.client_num))
    for a_iter in range(start_epoch, args.iters):
        # set global lr
        global_lr = args.lr if lr_sch is None else lr_sch.step()
        wandb.log({'global lr': global_lr}, commit=False)

        ##get global fc
        global_fc = fed.get_global_fc()
        global_model.load_state_dict(fed.model_accum.server_state_dict)



        ##train central generator
        if args.use_external == 'gen_inverse':
            for k, v in user_classifier.state_dict().items():
                if 'fc.weight' in k:
                    user_classifier.state_dict()[k].copy_(global_fc[0])
                if 'fc.bias' in k:
                    user_classifier.state_dict()[k].copy_(global_fc[1])
            Central_gen.train_generator(args, user_classifier)

        # ----------- Train Client ---------------
        train_loss_mt = AverageMeter()
        epsilon_mt = AverageMeter()
        best_alpha_mt = AverageMeter()
        print("============ Train epoch {} ============".format(a_iter))
        for client_idx in fed.client_sampler.iter():
            start_time = time.process_time()
            running_model = Running_model[client_idx]
            fed.download(running_model, client_idx)

            #prepare for VOS
            num_classes = fed.num_classes
            data_dim = 128
            if args.model == 'preresnet18':
                data_dim = 512
            elif args.model == 'alex':
                data_dim = int(4096*args.width_scale)
            data_dict = torch.zeros(num_classes, args.sample_number, data_dim).cuda()
            number_dict = {}
            for i in range(num_classes):
                number_dict[i] = 0
            eye_matrix = torch.eye(data_dim, device='cuda')

            fc_head = running_model.get_fc()
            fc_para = list(fc_head.parameters())
            optimizer_fc = torch.optim.SGD(
                fc_para, global_lr, momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)
            local_para = list(running_model.parameters())

            for i in range(len(fc_para)):
                for j in range(len(local_para)):
                    if fc_para[i].equal(local_para[j]):
                        local_para.pop(j)
                        break

            #print("para numnber: local para {}, fc para {}, rest para {}".format(len(list(running_model.parameters())), len(list(fc_head.parameters())), len(local_para)))
            optimizer_local = torch.optim.SGD(
                local_para + list(weight_energy[client_idx].parameters()) + \
                list(userlogistic[client_idx].parameters()), global_lr, momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)


            optimizer = torch.optim.SGD(
                list(running_model.parameters()) + list(weight_energy[client_idx].parameters()) + \
                list(userlogistic[client_idx].parameters()), global_lr, momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)
            #train for VOS
            if args.partition_mode != 'uni':
                max_iter = mean_batch_iters * args.wk_iters
            else:
                max_iter = len(train_loaders[client_idx]) * args.wk_iters
            if args.use_external == 'None':
                if (args.fl == 'fedprox') and (a_iter > start_epoch):
                    train_loss, total_iter[client_idx] = VOS_train_prox(user_class[client_idx], args.model,
                                                                   total_iter[client_idx], state, max_iter, global_model,
                                                                   running_model, train_loaders[client_idx],
                                                                   fed.num_classes, number_dict, args.sample_number,
                                                                   args.start_iter, data_dict,
                                                                   eye_matrix, userlogistic[client_idx], optimizer,
                                                                   args.loss_weight, weight_energy[client_idx],
                                                                   args.sample_from, args.select, verbose=args.verbose)
                else:
                    train_loss, total_iter[client_idx] = VOS_train(user_class[client_idx], args.model, total_iter[client_idx], state, max_iter, running_model, train_loaders[client_idx], fed.num_classes, number_dict, args.sample_number, args.start_iter, data_dict,
                      eye_matrix, userlogistic[client_idx], optimizer, args.loss_weight,  weight_energy[client_idx], args.sample_from, args.select, verbose=args.verbose)
            elif args.use_external == 'dataset':
                train_loss, total_iter[client_idx] = VOS_train2(args.model, total_iter[client_idx], state, max_iter, running_model, train_loaders[client_idx], train_loader_out, fed.num_classes, number_dict, args.sample_number, args.start_iter, data_dict,
                      eye_matrix, userlogistic[client_idx], optimizer, args.loss_weight,  weight_energy[client_idx], args.sample_from, args.select, verbose=args.verbose)
            elif args.use_external == 'class':
                train_loss, total_iter[client_idx] = VOS_train2(args.model, total_iter[client_idx], state, max_iter,
                                                                running_model, train_loaders[client_idx],
                                                                external_loader[client_idx], fed.num_classes, number_dict,
                                                                args.sample_number, args.start_iter, data_dict,
                                                                eye_matrix, userlogistic[client_idx], optimizer,
                                                                args.loss_weight, weight_energy[client_idx], args.sample_from,
                                                                args.select, verbose=args.verbose)

            elif args.use_external == 'gen_inverse':
                if args.select_generator != None:
                    if (args.fl == 'fedprox') and (a_iter > start_epoch):
                        train_loss, total_iter[client_idx] = topk_inversion_train_prox(num_classes, number_dict, data_dict,
                                                                                  args.sample_number, eye_matrix,
                                                                                  args.sample_from,
                                                                                  Central_gen.generative_model,
                                                                                  total_iter[client_idx],
                                                                                  user_class[client_idx], args.score,
                                                                                  args.m_in,
                                                                                  args.m_out, a_iter,
                                                                                  user_classifier,
                                                                                  total_iter[client_idx], state,
                                                                                  max_iter, global_model,
                                                                                  running_model,
                                                                                  train_loaders[client_idx], optimizer,
                                                                                  verbose=args.verbose,
                                                                                  logistic_regression=userlogistic[
                                                                                      client_idx],
                                                                                  weight_energy=weight_energy[
                                                                                      client_idx],
                                                                                  select=args.select_generator,
                                                                                  soft=args.soft,
                                                                                  optimizer_fc=optimizer_fc,
                                                                                  optimizer_local=optimizer_local)
                    else:
                        train_loss, total_iter[client_idx] = topk_inversion_train(num_classes, number_dict, data_dict,
                                                                              args.sample_number, eye_matrix,
                                                                              args.sample_from,
                                                                              Central_gen.generative_model,
                                                                              total_iter[client_idx],
                                                                              user_class[client_idx], args.score,
                                                                              args.m_in,
                                                                              args.m_out, a_iter,
                                                                              user_classifier,
                                                                              total_iter[client_idx], state, max_iter,
                                                                              running_model,
                                                                              train_loaders[client_idx], optimizer,
                                                                              verbose=args.verbose,
                                                                              logistic_regression=userlogistic[
                                                                                  client_idx],
                                                                              weight_energy=weight_energy[client_idx],
                                                                              select=args.select_generator,
                                                                              soft=args.soft, optimizer_fc=optimizer_fc, optimizer_local=optimizer_local)
                if args.score == 'energy_VOS':
                    train_loss, total_iter[client_idx] = inversion_train(Central_gen.generative_model,
                                                                         total_iter[client_idx],
                                                                         user_class[client_idx], args.score,
                                                                         args.m_in,
                                                                         args.m_out, a_iter,
                                                                         user_classifier, args.oe_batch_size,
                                                                         total_iter[client_idx], state, max_iter,
                                                                         running_model,
                                                                         train_loaders[client_idx], optimizer,
                                                                         verbose=args.verbose,
                                                                         logistic_regression=userlogistic[
                                                                             client_idx],
                                                                         weight_energy=weight_energy[client_idx])
                else:
                    train_loss, total_iter[client_idx] = inversion_train(Central_gen.generative_model,
                                                                         total_iter[client_idx],
                                                                         user_class[client_idx],
                                                                         args.score, args.m_in, args.m_out, a_iter,
                                                                         user_classifier, args.oe_batch_size,
                                                                         total_iter[client_idx], state, max_iter,
                                                                         running_model,
                                                                         train_loaders[client_idx], optimizer,
                                                                         verbose=args.verbose, )



            # Upload
            fed.upload(running_model, client_idx)

            # Log
            client_name = fed.clients[client_idx]
            elapsed = time.process_time() - start_time
            wandb.log({f'{client_name}_train_elapsed': elapsed}, commit=False)
            train_elapsed[client_idx].append(elapsed)

            train_loss_mt.append(train_loss)


            print(f' User-{client_name:<10s} Train | Loss: {train_loss:.4f} |'
                  f' Elapsed: {elapsed:.2f} s')

            wandb.log({
                f"{client_name} train_loss": train_loss,
            }, commit=False)

        # Use accumulated model to update server model
        fed.aggregate()

        # ----------- Validation ---------------
        val_acc_list, val_loss = fed_test(fed, running_model, val_loaders, args.verbose)
        if args.adv_lmbd > 0:
            print(f' Avg Val SAcc {np.mean(val_acc_list) * 100:.2f}%')
            wandb.log({'val_sacc': np.mean(val_acc_list)}, commit=False)
            val_racc_list, val_rloss = fed_test(fed, running_model, val_loaders, args.verbose,
                                                adversary=adversary)
            print(f' Avg Val RAcc {np.mean(val_racc_list) * 100:.2f}%')
            wandb.log({'val_racc': np.mean(val_racc_list)}, commit=False)

            val_acc_list = [(1 - args.adv_lmbd) * sa_ + args.adv_lmbd * ra_
                            for sa_, ra_ in zip(val_acc_list, val_racc_list)]
            val_loss = (1 - args.adv_lmbd) * val_loss + args.adv_lmbd * val_rloss

        # Log averaged
        print(f' [Overall] Train Loss {train_loss_mt.avg:.4f} '
              f' | Val Acc {np.mean(val_acc_list) * 100:.2f}%')
        wandb.log({
            f"train_loss": train_loss_mt.avg,
            f"val_loss": val_loss,
            f"val_acc": np.mean(val_acc_list),
        }, commit=False)


        # ----------- Save checkpoint -----------
        if np.mean(val_acc_list) > np.mean(best_acc):
            best_epoch = a_iter
            for client_idx in range(fed.client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
                if args.verbose > 0:
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                        fed.clients[client_idx], best_epoch, best_acc[client_idx]))
            print(' [Best Val] Acc {:.4f}'.format(np.mean(val_acc_list)))

            # Save
            print(f' Saving the local and server checkpoint to {SAVE_FILE}')
            save_dict = {
                'server_model': fed.model_accum.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'a_iter': a_iter,
                'all_domains': fed.all_domains,
                'train_elapsed': train_elapsed,
            }
            if args.visualization == False:
                torch.save(save_dict, SAVE_FILE)
        wandb.log({
            f"best_val_acc": np.mean(best_acc),
        }, commit=True)
    if args.visualization:
        print("Start visualization!")
        global_fc = fed.get_global_fc()
        for k, v in user_classifier.state_dict().items():
            if 'fc.weight' in k:
                user_classifier.state_dict()[k] = global_fc[0]
            if 'fc.bias' in k:
                user_classifier.state_dict()[k] = global_fc[1]
        Central_gen = CentralGen(args, 10000, fed.num_classes, model=args.model)
        Central_gen.train_generator(args, user_classifier)
        Central_gen2 = CentralGen(args, 10000, fed.num_classes, model=args.model)
        Central_gen2.generative_model.load_state_dict(get_weights(Central_gen.generative_model))
        for client_idx in fed.client_sampler.iter():
            num_classes = fed.num_classes
            data_dim = 128
            if args.model == 'preresnet18':
                data_dim = 512
            data_dict = torch.zeros(num_classes, args.sample_number, data_dim).cuda()
            number_dict = {}
            for i in range(num_classes):
                number_dict[i] = 0
            eye_matrix = torch.eye(data_dim, device='cuda')
            fed.download(running_model, client_idx)
            #_, test_acc = test(running_model, client_idx, loss_fun, device,
            #                   adversary=adversary)
            #print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[client_idx], test_acc))
            from nets.profile_func import count_params_by_state
            gen_params = count_params_by_state(Central_gen2.generative_model)

            print('gen_model size: %.4fMB' % (gen_params / 1e6))
            params = count_params_by_state(running_model)
            print('model size: %.4fMB' % (params / 1e6))
            visualization(client_idx, num_classes, number_dict, data_dict, args.sample_number, eye_matrix,
                          100000,  Central_gen2.generative_model,
                          total_iter[client_idx],
                          user_class[client_idx],
                         user_classifier,
                         total_iter[client_idx], state, max_iter,
                          running_model,
                          train_loaders[client_idx],
                          verbose=args.verbose,
                          logistic_regression=userlogistic[
                              client_idx],
                          weight_energy=weight_energy[client_idx], select=1, soft=args.soft)
            #visualization_external(client_idx, num_classes, number_dict, data_dict, args.sample_number, eye_matrix,
            #              100000,  Central_gen2.generative_model,
            #              total_iter[client_idx],
            #              user_class[client_idx],
            #              user_classifier,
            #              total_iter[client_idx], state, max_iter,
            #              running_model,
            #              train_loaders[client_idx],
            #              verbose=args.verbose,
            #              logistic_regression=userlogistic[
            #                  client_idx],
            #              weight_energy=weight_energy[client_idx], select=1, soft=args.soft, external_loader=external_loader[client_idx])
