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
        # read image as data
        real = functions.read_image(opt)
        # adjust size 
        functions.adjust_scales2image(real, opt)
        # train model based on given args
        train(opt, Gs, Zs, reals, NoiseAmp)
        # generate new image based on trained model
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
