python train_model.py -c=configs/training/targets/ResNet18_FaceScrub_midre.yaml

CUDA_VISIBLE_DEVICES=0 python attack.py -c=configs/attacking/FaceScrub_FFHQ_resnet18_midre.yml --start_id=0

# git add .
# git commit -m "Initial commit"
# git push origin main

