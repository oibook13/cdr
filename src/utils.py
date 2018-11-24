import os
import torch
import torch.utils.data as data
import shutil
from PIL import Image

def args_str2list(s):
    try: 
        lps = list(map(float, s.split(',')))
        return lps
    except:
        raise argparse.ArgumentTypeError('Lps inputs must be float')

def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0,len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames,classes

def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            #if datasets.folder.is_image_file(fname):
            if datasets.folder.has_file_allowed_extension(fname,['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images

class TImgNetDataset(data.Dataset):
    """Tiny ImageNet val set."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        #self.imgs = load_allimages(img_path)
        #self.imgs.sort(key = lambda t: t[0])
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

    def __len__(self):
        return len(self.imgs)

class AverageMeter(object):
    """
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth'))

def adjust_lr(optimizer, epoch, lr_decay, lr_scheduler):
    if epoch in lr_scheduler:
        tmp_lr = list(optimizer.param_groups)[0]['lr'] * lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = tmp_lr
