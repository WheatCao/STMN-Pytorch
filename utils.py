import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print (P)

    return y_train, actual_noise,P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
#        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
#    print (P)

    return y_train, actual_noise,P


def noisify_multiclass_asymmetric_songhua(y_train, noise, random_state=None, n_classes=10):

    np.random.seed(1)
    t = np.random.rand(n_classes, n_classes)
    i = np.eye(n_classes)
    if noise == 0.1:
        t = t + 3.0 * n_classes * i
    if noise == 0.3:
        t = t + 1.2 * n_classes * i
    for a in range(n_classes):
        t[a] = t[a] / t[a].sum()
    y_train_noisy = multiclass_noisify(y_train, P=t, random_state=random_state)
    actual_noise = 0.3
    y_train = y_train_noisy
    print(t)

    return y_train, actual_noise, t


def noisify_multiclass_asymmetric_mnist(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.eye(10)
    n = noise

    P[7, 7], P[7, 1] = 1. - n, n
    # 2 -> 7
    P[2, 2], P[2, 7] = 1. - n, n

    # 5 <-> 6
    P[5, 5], P[5, 6] = 1. - n, n
    P[6, 6], P[6, 5] = 1. - n, n

    # 3 -> 8
    P[3, 3], P[3, 8] = 1. - n, n

    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)

    y_train = y_train_noisy
    print (P)

    return y_train, actual_noise,P

def noisify_multiclass_asymmetric_fashionmnist(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.eye(10)
    n = noise
    # 0 -> 6
    P[0, 0], P[0, 6] = 1. - n, n
    # 2 -> 4
    P[2, 2], P[2, 4] = 1. - n, n

    # 5 <-> 7
    P[5, 5], P[5, 7] = 1. - n, n
    P[7, 7], P[7, 5] = 1. - n, n

    # 3 -> 8
    #P[3, 3], P[3, 8] = 1. - n, n

    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)

    y_train = y_train_noisy
    print (P)

    return y_train, actual_noise,P


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

def noisify_multiclass_asymmetric_cifar10(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    source_class = [9, 2, 3, 5, 4]
    target_class = [1, 0, 5, 3, 7]
    y_train_ = y_train
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(np.array(y_train) == s)[0]
        n_noisy = int(noise * cls_idx.shape[0])
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        for idx in noisy_sample_index:
            y_train_[idx] = t
    return y_train_, source_class, target_class

def noisify_multiclass_asymmetric_cifar100(y_train, noise, random_state=None, nb_classes=100):
    """mistakes:
        flip in the symmetric way
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

            y_train_noisy = multiclass_noisify(np.array(y_train), P=P, random_state=random_state)
            actual_noise = (y_train_noisy != np.array(y_train)).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        targets = y_train_noisy
    return targets, actual_noise, P





def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate
import torch.nn as nn

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


import torch
import torch.nn.functional as F
def gaussian_window(size, sigma):
    """
    创建一个高斯窗口，用于SSIM中的局部区域加权。
    """
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=3, sigma=1.5, size_average=True):
        """
        初始化SSIM损失函数模块。
        """
        super(SSIMLoss, self).__init__()
        self.window = gaussian_window(window_size, sigma).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.size_average = size_average
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, img1, img2):
        """
        计算批量矩阵对的SSIM损失。
        """
        mu1 = F.conv2d(img1, self.window, padding=self.window.shape[2] // 2, groups=1)
        mu2 = F.conv2d(img2, self.window, padding=self.window.shape[2] // 2, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window.shape[2] // 2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window.shape[2] // 2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window.shape[2] // 2, groups=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        ssim_map = 1 - ssim_map  # 将SSIM转换为非负损失

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class StructuralSimilarityLoss(torch.nn.Module):
    def __init__(self, window_size=10, sigma=1.5, size_average=True, alpha=0.9):
        """
        初始化混合损失函数模块。
        """
        super(StructuralSimilarityLoss, self).__init__()
        self.ssim_window = gaussian_window(window_size, sigma).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.size_average = size_average
        self.alpha = alpha  # 混合系数，用于平衡SSIM损失和MSE损失
        self.C1 = 0.01**2

    def ssim(self, img1, img2):
        """
        计算图像之间的结构相似性。
        """
        mu1 = F.conv2d(img1, self.ssim_window, padding=self.ssim_window.shape[2] // 2, groups=1)
        mu2 = F.conv2d(img2, self.ssim_window, padding=self.ssim_window.shape[2] // 2, groups=1)

        sigma1_sq = F.conv2d(img1 * img1, self.ssim_window, padding=self.ssim_window.shape[2] // 2, groups=1) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2 * img2, self.ssim_window, padding=self.ssim_window.shape[2] // 2, groups=1) - mu2.pow(2)
        sigma12 = F.conv2d(img1 * img2, self.ssim_window, padding=self.ssim_window.shape[2] // 2, groups=1) - mu1 * mu2

        structural_similarity = sigma12 + self.C1 / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq)+ self.C1)

        return structural_similarity

    def forward(self, img1, img2):
        """
        计算混合损失，结合结构相似性和均方误差。
        """
        ssim_loss = 1 - self.ssim(img1, img2)
        mse_loss = F.mse_loss(img1, img2, reduction='mean')

        mixed_loss = (1 - self.alpha) * ssim_loss + self.alpha * mse_loss

        if self.size_average:
            return mixed_loss.mean()
        else:
            return mixed_loss.mean(1).mean(1).mean(1)