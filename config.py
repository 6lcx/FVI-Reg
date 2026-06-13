from Controllers import BaseController_Seg_GSFVI
from Networks import (SegFormer_GSFVIGenerativeNetwork)

config = {
    'GPUNo': '2',
    'mode': 'Train',
    'network': 'SegFormer_GSFVI',
    'name': 'mm',
    
    'dataset': {
        'training_list_path': '/root/dym/Dataset/dataset2D_MnMs/small/training_pair.txt',
        'testing_list_path': '/root/dym/Dataset/dataset2D_MnMs/small/testing_pair.txt',
        'validation_list_path': '/root/dym/code_dym/Dataset/dataset2D_MnMs/small/validation_pair.txt',
        'pair_dir': '/root/dym/code_dym/Dataset/dataset2D_MnMs/data/',
        'resolution_path': '/root/dym/Dataset/dataset2D_MnMs/resolution.txt'
    },
    
    'Train': {
        'batch_size': 16 ,
        'model_save_dir':
        '/root/dym/train_result/',
        'lr': 3e-4,
        'start_epoch': 0,
        'weight_decay': 0,
        'max_epoch': 4000,
        'save_checkpoint_step': 500,
        'v_step': 10,

        'earlystop': {
            'min_delta': 0.00001,
            'patience': 500
        },
    },
    'Test': {
        'epoch': 'best',
        'model_save_path': 'None',
        'excel_save_path': 'None',
        'verbose': 2,
    },
    'SpeedTest': {
        'epoch': 'best',
        'model_save_path': 'None',
        'device': 'cpu'
    },
    'Hyperopt': {
        'n_trials': 5,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 500
        },
        'max_epoch': 2000,
        'lr': 3e-4
    },

    'SegFormer_GSFVI': {
        'controller': BaseController_Seg_GSFVI,
        'network': SegFormer_GSFVIGenerativeNetwork,
        'params': {
            'encoder_param': {
                'mit_embed_dims':[32, 64, 160, 256],
                 'mit_depths':[2, 2, 2, 2],
                 'mit_num_heads':[1, 2, 5, 8],
                 'mit_sr_ratios':[8, 4, 2, 1],
                 'in_chans':2,
                 'drop_rate':0.0,
                 'drop_path_rate':0.0,
            },
            'c': 4.0,
            'i_size': [128, 128],
            'similarity_factor': 120000,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            },
            'prior_mean_source': 'teacher',
            'prior_feature_source': 'f3',
            'prior_cov_source': 'dkl',
        }
    },
}
