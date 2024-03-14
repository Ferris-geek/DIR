CUDA_VISIBLE_DEVICES=0 python scripts/rgb2ir.py --steps 200 \
--indir ./dataset/Freiburg/visible \
--outdir ./results/ldm_frei_rectev_s200 \
--config ./configs/latent-diffusion/freiburg256-vqf4_rec_tev.yaml \
--checkpoint path/to/checkpoint