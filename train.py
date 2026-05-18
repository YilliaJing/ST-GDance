from STGDance import STGDance
from args import parse_train_opt
import torch
import wandb

torch.cuda.set_device(1)

def train(opt):
    wandb.init(
        project=opt.wandb_pj_name,  # 你的项目名称
        name=opt.exp_name,  # 当前运行名称
        config=vars(opt),  # 记录所有超参数
    )

    model = STGDance(feature_type=opt.feature_type, learning_rate=opt.learning_rate)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
