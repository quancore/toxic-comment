import torch
from metrics_loss import LabelSmoothing, FocalLosswSmooting
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


class TrainGlobalConfig:
    num_classes = 2
    num_workers = 0 
    batch_size = 16 
    n_epochs = 4
    # number of epoch validation set is used for finetuning trained model
    n_val_epoch = 3
    lr = 0.1*1e-5
    warmup = 200

    # -------------------
    verbose = True
    verbose_step = 50
    # -------------------

    # --------------------
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss
    # SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler_params = dict(
    #     mode='max',
    #     factor=0.7,
    #     patience=0,
    #     verbose=False, 
    #     threshold=0.0001,
    #     threshold_mode='abs',
    #     cooldown=0, 
    #     min_lr=1e-8,
    #     eps=1e-08
    # )
    # SchedulerClass = torch.optim.lr_scheduler.CyclicLR
    # scheduler_params = dict(
    #     base_lr=lr,
    #     max_lr=1e-4,
    #     step_size_up=2000, 
    #     step_size_down=None, 
    #     mode='triangular2', 
    #     gamma=1.0, 
    #     scale_fn=None, 
    #     scale_mode='cycle', 
    #     cycle_momentum=False, 
    #     base_momentum=0.8, 
    #     max_momentum=0.9, 
    # )
    SchedulerClass = get_cosine_with_hard_restarts_schedule_with_warmup
    scheduler_params = dict(
        num_warmup_steps = warmup,
        num_training_steps = 2000
    )
    # ------------------
    
    # -------------------
    # criterion = FocalLosswSmooting()
    criterion = LabelSmoothing()
    # -------------------
    
    #--------------------
    pseudolabeling_threshold = 0.3
    #--------------------
    
    #--------------------
    # this parameter controls whether we are using shocastic weight avaraging or not
    use_SWA = True
    # this parameter controls whether we are using different lr for backbone model and classifier head
    use_diff_lr = False
    #-------------------