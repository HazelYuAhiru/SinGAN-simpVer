import os
import functions
from manipulate import SinGAN_generate
from training import train
from config import get_arguments


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='../Input/Images')
    parser.add_argument('--input_name', help='input image name', default='image.png')
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    #
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        # 将图片读取成torch版的数据
        real = functions.read_image(opt)
        # 将图片适配尺寸
        functions.adjust_scales2image(real, opt)
        # 开始训练模型 opt 手动输入的参数
        train(opt, Gs, Zs, reals, NoiseAmp)
        # 根据模型生成图片  生成具有任意大小和比例的新图像
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
