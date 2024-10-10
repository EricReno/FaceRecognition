import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FaceNet')

    parser.add_argument('--cuda', 
                        default=True,   
                        type=bool,
                        help='Enable CUDA for GPU acceleration.')
    parser.add_argument('--seed', 
                        default=1234,   
                        help='Enable CUDA for GPU acceleration.')
    parser.add_argument('--num_workers',  
                        default=1,
                        type=int,
                        help='Number of CPU threads to use during data loading.')
    
    # Dataset settings
    parser.add_argument('--data_root',
                        default='/data/CASIA-FaceV5_CUT',
                        type=str,
                        help='Root directory of the dataset.')
    
    parser.add_argument('--image_set',
                        default=0.9,
                        help='ratio used for training or val.')
    
    # Model settings
    parser.add_argument('--backbone', 
                        default='mobilenet_v1',
                        type=str,
                        choices=['mobilenet_v1', 'inception_v1'],
                        help='Backbone network architecture.')
    
    parser.add_argument('--image_size',
                        default=160,
                        type=int,
                        help='Input image size.')
    
    # Training settings
    parser.add_argument('--batch_size',
                        default=96,
                        type=int,
                        help='Batch size used during training (per GPU).')
    
    parser.add_argument('--epochs_total',
                        default=100,
                        type=int,
                        help='Total number of training epochs.')
    
    parser.add_argument('--warmup_epochs',
                        default=3,
                        type=int,
                        help='Number of warm-up epochs.')
    
    parser.add_argument('--save_checkpoint_epoch',
                        default=0,
                        type=int,
                        help='Epoch interval to save model checkpoints.')
    
    # Optimizer settings
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Base learning rate.')
    
    parser.add_argument('--lr_scheduler',             
                        default='linear',
                        type=str,
                        help='Base learning rate.')
    
    parser.add_argument('--grad_accumulate', 
                        default=1, type=int,
                        help='gradient accumulation')
    
    parser.add_argument('--learning_rate',             
                        default=0.01,
                        type=float,
                        help='Base learning rate.')
    
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum factor for SGD optimizer.')
    
    parser.add_argument('--weight_decay',
                        default=0.0005,
                        type=float,
                        help='Weight decay factor for regularization.')
    
    # Model checkpoint
    parser.add_argument('--model_weight_path',         
                        default='best.pth',                
                        type=str,
                        help='Path to the initial model weights.')

    parser.add_argument('--resume_weight_path',         
                        default='best.pth',                
                        type=str,
                        help='Path to the checkpoint from which to resume training.')
    
    parser.add_argument('--eval_visualization',         
                        default=False,                
                        type=bool,
                        help='Whether to visualize the evaluation results.')

    return parser.parse_args()