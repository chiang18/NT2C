from utils.config import parse_args
from utils.data_loader_128 import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
#from models.wgan_gradient_penalty import WGAN_GP
from models.wgan_gradient_penalty_savemrc_1024 import WGAN_GP

def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)
    # 获取第一个批次的数据
    first_batch = next(iter(train_loader))  # 使用 train_loader 或 test_loader
    # 获取第一个批次的输入和目标数据
    inputs, targets = first_batch
    # 输出第一个批次的形状
    print("First Batch Input Shape:", inputs.shape)
    print("First Batch Target Shape:", targets.shape)
    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        #for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)