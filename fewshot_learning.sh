gpus=0 #,1,2,3

# Meta-Baseline
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metabaseline_in22k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metabaseline_cliplaion2b_notreset_lr=1.0e-4_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metabaseline_cliplaion2b_notreset_lr=1.0e-4_wcap_freeze2q_lc=0.1

# MetaOptNet
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metaoptnet_scratch_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metaoptnet_in1k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metaoptnet_in22k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metaoptnet_cliplaion2b_notreset_lr=1.0e-5_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name metaoptnet_cliplaion2b_notreset_lr=1.0e-5_wcap_notfreeze2q_lc=0.1

# ProtoNet
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name protonet_scratch_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name protonet_in1k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name protonet_in22k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name protonet_cliplaion2b_notreset_lr=3.0e-5_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name protonet_cliplaion2b_notreset_lr=3.0e-5_wcap_freeze2q_lc=1.0

# SNAIL
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name snail_scratch_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name snail_in1k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name snail_in22k_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name snail_cliplaion2b_notreset_lr=1.0e-4_wocap
CUDA_VISIBLE_DEVICES=$gpus python main_bongard.py --config-name snail_cliplaion2b_notreset_lr=1.0e-4_wcap_notfreeze2q_lc=1.0