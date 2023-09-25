# this script is used to train biADT c2f
nproc_per_node=4
batch_size=1

coco_path=../../../datasets

output_dir=biADT8_r50_20230829_4gpu
epochs=50
lr_drop=40
num_queries=300
backbone=resnet50
resume_weights=${output_dir}/checkpoint0044_beforedrop.pth
dataset_file=cityscapes
lr=2e-4
lr_backbone=2e-5

tgt0_xy=0.075     
tgt0_zd=0.025
tgt0_mi=1.0e-8
srcs_xy=0.070
srcs_zd=0.035     
tgts_xy=0.050
tgts_zd=0.025
srcs_mi=1.0e-7
tgts_mi=1.0e-6
bkbs_xy=0.100
bkbs_zd=0.010
bkbs_mi=1.0e-4
src6_xy=0.100       
src6_zd=0.010  
tgt5_xy=0.100       
tgt5_zd=0.010   
src6_mi=5.0e-5      
tgt5_mi=5.0e-5      
tgts_margin=0.50
srcs_margin=0.50
dropout=0.0
#random_refpoints_xy=1
with_aqt=0
space_q=0.10
chann_q=0.001
insta_q=0.001

CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --master_port 10038 --nproc_per_node=${nproc_per_node} \
        main.py -m dab_deformable_detr \
        --output_dir ${output_dir}        \
        --dataset_file ${dataset_file}    \
        --backbone ${backbone}            \
        --batch_size ${batch_size}        \
        --lr ${lr}                        \
        --lr_drop ${lr_drop}              \
        --lr_backbone ${lr_backbone}      \
        --num_queries ${num_queries}      \
        --epochs $epochs                  \
        --coco_path ${coco_path}          \
        --hidden_dim 384                  \
        --enc_layers 6                    \
        --bkbs_da_loss_xy ${bkbs_xy}      \
        --bkbs_da_loss_zd ${bkbs_zd}      \
        --srcs_da_loss_xy ${srcs_xy}      \
        --srcs_da_loss_zd ${srcs_zd}      \
        --tgts_da_loss_xy ${tgts_xy}      \
        --tgts_da_loss_zd ${tgts_zd}      \
        --src6_da_loss_xy ${src6_xy}      \
        --src6_da_loss_zd ${src6_zd}      \
        --tgt0_da_loss_xy ${tgt0_xy}      \
        --tgt0_da_loss_zd ${tgt0_zd}      \
        --tgt5_da_loss_xy ${tgt5_xy}      \
        --tgt5_da_loss_zd ${tgt5_zd}      \
        --srcs_mi_loss ${srcs_mi}         \
        --tgts_mi_loss ${tgts_mi}         \
        --bkbs_mi_loss ${bkbs_mi}         \
        --src6_mi_loss ${src6_mi}         \
        --tgt5_mi_loss ${tgt5_mi}         \
        --tgt0_mi_loss ${tgt0_mi}         \
        --margin_src ${srcs_margin}       \
        --margin_tgt ${tgts_margin}       \
        --with_aqt ${with_aqt}            \
        --space_q ${space_q}              \
        --chann_q ${chann_q}              \
        --insta_q ${insta_q}              \
        --training_phase wa_src_tgt       \
        --dropout ${dropout}              \
        --random_refpoints_xy             \
       