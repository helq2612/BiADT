# used for inference on c2fc, with AQT
nproc_per_node=4
batch_size=1

coco_path=../datasets
output_dir=biadt_r50_4gpu8batch_20230129
epochs=50
lr_drop=40
num_queries=300
backbone=resnet50
resume_weights=../weights/c2fc_aqt_biadt.pth
dataset_file=cityscapes
with_aqt=1
space_q=0.10
chann_q=0.001
insta_q=0.001

python -m torch.distributed.launch --master_port 10038 --nproc_per_node=${nproc_per_node} \
        main.py -m dab_deformable_detr \
        --output_dir ${output_dir}      \
        --backbone ${backbone}          \
        --batch_size ${batch_size}      \
        --lr 5e-5                       \
        --lr_drop ${lr_drop}            \
        --num_queries ${num_queries}    \
        --epochs $epochs                \
        --coco_path ${coco_path}        \
        --hidden_dim 384                \
        --resume ${resume_weights}      \
        --with_aqt ${with_aqt}          \
        --space_q ${space_q}            \
        --chann_q ${chann_q}            \
        --insta_q ${insta_q}            \
        --eval                          \
        --dataset_file ${dataset_file}  \