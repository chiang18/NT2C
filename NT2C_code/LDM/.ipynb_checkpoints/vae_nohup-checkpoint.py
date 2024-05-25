#!/usr/bin/env python
# coding: utf-8

import subprocess

# command = '''accelerate launch --num_processes=4 --num_machines=1 \
# /home/m112040012/vscode/LDM/diffusers/examples/vae/train_vae.py \
# --mixed_precision='no' \
# --resolution=256 \
# --num_train_epochs=150 \
# --validation_epochs=1000 \
# --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
# --train_data_dir="/home/m112040012/vscode/data/10077/noise_patches_jpg/" \
# --output_dir="/home/m112040012/vscode/wgan/LDM/vae/" \
# --train_batch_size=2 \
# --gradient_accumulation_steps=1 \
# --checkpointing_steps=100000 \
# --scale_lr \
# --checkpoints_total_limit=5 \
# --logging_dir="/home/m112040012/vscode/wgan/LDM/vae/log/" \
# --lr_warmup_steps=500 \
# --learning_rate=1e-3 \
# --lr_scheduler="cosine" '''



# command = '''accelerate launch --num_processes=4 --num_machines=1 \
# /home/m112040012/vscode/LDM/diffuser_VQmodel/examples/vqgan/train_vqvae.py \
# --mixed_precision='no' \
# --resolution=256 \
# --num_train_epochs=100 \
# --validation_epochs=1000 \
# --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
# --train_data_dir="/home/m112040012/vscode/data/10077/noise_patches_jpg/" \
# --output_dir="/home/m112040012/vscode/vae_model/vqvae2/" \
# --train_batch_size=4 \
# --gradient_accumulation_steps=1 \
# --checkpointing_steps=100000 \
# --scale_lr \
# --checkpoints_total_limit=5 \
# --logging_dir="/home/m112040012/vscode/vae_model/vqvae2/log/" \
# --lr_warmup_steps=500 \
# --learning_rate=1e-3 \
# --lr_scheduler="linear" '''

command = '''CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 --num_machines=1 \
/home/m112040012/vscode/LDM/diffuser_VQmodel/examples/vqgan/train_vqgan.py \
--mixed_precision='no' \
--resolution=256 \
--num_train_epochs=60 \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--train_data_dir="/home/m112040012/vscode/data/10077/noise_patches_jpg/" \
--output_dir="/home/m112040012/vscode/vae_model/vqvae3/" \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--checkpointing_steps=2000000 \
--scale_lr \
--checkpoints_total_limit=1 \
--logging_dir="/home/m112040012/vscode/vae_model/vqvae3/log/" \
--lr_warmup_steps=700 \
--discr_learning_rate=1e-4 \
--learning_rate=1e-3 \
--use_8bit_adam \
--discr_lr_scheduler="constant" \
--lr_scheduler="linear" '''

#--gradient_checkpointing \
#RuntimeError: Given groups=1, weight of size [128, 3, 3, 3], expected input[2, 1, 256, 256] to have 3 channels, but got 1 channels instead

subprocess.run(command, shell=True)
