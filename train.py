import os
import time
import torch
import numpy
from torch.utils.tensorboard import SummaryWriter

from config import parse_args
from evaluate import build_eval
from model.build import build_facenet
from utils.flops import compute_flops
from utils.criterion import build_loss
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lambda_lr_scheduler
from dataset.build import build_dataset, build_dataloader

def train():
    args = parse_args()
    writer = SummaryWriter('deploy')
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---------------------------- Build --------------------------
    num_classes = len(os.listdir(args.data_root))

    val_dataset = build_dataset(args, is_train=False, num_classes=num_classes)
    train_dataset = build_dataset(args, is_train=True, num_classes=num_classes)
    train_dataloader = build_dataloader(args, train_dataset)

    model = build_facenet(args, device, num_classes)
    compute_flops(model, args.image_size, device)
          
    loss =  build_loss(args, device, 'triplet_loss')
    
    evaluator = build_eval(args, val_dataset, device)
    
    optimizer, start_epoch = build_optimizer(args, model)

    lr_scheduler, lf = build_lambda_lr_scheduler(args, optimizer)
    if args.resume_weight_path and args.resume_weight_path != 'None':
        lr_scheduler.last_epoch = start_epoch - 1
        optimizer.step()
        lr_scheduler.step()
    
    # ----------------------- Train --------------------------------
    print('==============================')
    max_Acc = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs_total+1):
        model.train()
        train_loss = 0.0
        model.trainable = True
        for iteration, (images, labels) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * len(train_dataloader)
            if epoch < args.warmup_epochs:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration,
                                           [0, args.warmup_epochs*len(train_dataloader)],
                                           [0.1 if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                  
            ## forward
            images = images.to(device)
            labels = labels.to(device)
            outputs1, outputs2 = model(images)

            ## loss
            _Triplet_loss = loss(outputs1, args.batch_size//3)
            _CE_loss = torch.nn.NLLLoss()(torch.nn.functional.log_softmax(outputs2, dim = -1), labels)
            _loss = _Triplet_loss + _CE_loss
            if args.grad_accumulate > 1:
               _loss /= args.grad_accumulate
               _CE_loss /= args.grad_accumulate
               _Triplet_loss /= args.grad_accumulate
            _loss.backward()
            
            # optimizer.step
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch [{}:{}/{}:{}, {}], lr: {:.5f}, Loss: {:8.4f}, Triplet_loss: {:8.4f}, CE_loss: {:6.3f}".format(
                epoch, args.epochs_total, iteration+1, len(train_dataloader),
                time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                optimizer.param_groups[2]['lr'], _loss, _Triplet_loss, _CE_loss))
            train_loss += _loss * images.size(0)

        lr_scheduler.step()

        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # save_model
        model.eval()
        model.trainable = False
        if epoch >= args.save_checkpoint_epoch:
            ckpt_path = os.path.join(os.getcwd(), 'deploy', 'best.pth')
            if not os.path.exists(os.path.dirname(ckpt_path)): 
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                Acc = evaluator.eval(model, epoch)
            writer.add_scalar('Acc', Acc, epoch)

            if Acc > max_Acc:
                torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'Acc':Acc,
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_Acc = Acc
        
if __name__ == "__main__":
    train()