import config, os
from utils import supervisor
import torch
from torchvision import datasets, transforms
from PIL import Image

class BackdoorDefense():
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == 'gtsrb':
            self.img_size = 32
            self.num_classes = 43
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar10':      
            self.img_size = 32
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar100':
            print('<To Be Implemented> Dataset = %s' % args.dataset)
            exit(0)
        elif args.dataset == 'imagenette':   
            self.img_size = 224
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'imagenet':
            self.img_size = 224
            self.num_classes = 1000
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        else:
            print('<Undefined> Dataset = %s' % args.dataset)
            exit(0)
        
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(args)
        
        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = config.target_class[args.dataset]
        self.device='cuda'

        if args.poison_type == 'SSDT':
            self.poison_transform = type('DummyTransform', (), {
                'transform': lambda self, x, y: (x, y)
            })()
        else:
            self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                                target_class=config.target_class[args.dataset], trigger_transform=self.trigger_transform,
                                                                is_normalized_input=(not args.no_normalize),
                                                                alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                                trigger_name=args.trigger, args=args)
        
        if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
            self.source_classes = [config.source_class]
        else:
            self.source_classes = None

        if args.poison_type != 'SSDT':
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_path = os.path.join(config.triggers_dir, args.trigger)
            print('trigger_path:', trigger_path)
            self.trigger_mark = Image.open(trigger_path).convert("RGB")
            self.trigger_mark = trigger_transform(self.trigger_mark).cuda()

            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % args.trigger)
            if os.path.exists(trigger_mask_path): # if there explicitly exists a trigger mask (with the same name)
                print('trigger_mask_path:', trigger_mask_path)
                self.trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                self.trigger_mask = transforms.ToTensor()(self.trigger_mask)[0].cuda() # only use 1 channel
            else: # by default, all black pixels are masked with 0's (not used)
                print('No trigger mask found! By default masking all black pixels...')
                self.trigger_mask = torch.logical_or(torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0), self.trigger_mark[2] > 0).cuda()

        # ... (phần đầu của __init__ không đổi) ...

        self.poison_set_dir = supervisor.get_poison_set_dir(args)

        # Xác định đường dẫn model dựa trên loại độc hại
        if args.poison_type == 'SSDT':
            # Sử dụng dấu gạch chéo xuôi để tránh lỗi escape characters
            model_path = 'checkpoints/cifar10/SSDT/target_0/SSDT_cifar10_ckpt.pth.tar'
        else:
            model_path = supervisor.get_model_dir(args)

        arch = supervisor.get_arch(args)
        self.model = arch(num_classes=self.num_classes)

        if os.path.exists(model_path):
            print(f"Loading model '{model_path}'...")
            # Tải checkpoint với thiết bị tương ứng
            checkpoint = torch.load(model_path, map_location=self.device)

            if args.poison_type == 'SSDT':
                # Kiểm tra xem checkpoint có chứa key 'netC' hay không
                if 'netC' in checkpoint:
                    self.model.load_state_dict(checkpoint['netC'], strict=False)
                    print("SSDT model loaded successfully using 'netC' weights.")
                else:
                    print("Warning: SSDT checkpoint does not contain 'netC'. Attempting full load.")
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model.load_state_dict(checkpoint)

            print("Model loaded.")
        else:
            print(f"Model '{model_path}' not found.")

        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.eval()

        
