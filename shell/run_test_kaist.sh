CUDA_VISIBLE_DEVICES=0 python scripts/rgb2ir.py --steps 200 \
--indir ./dataset/KAISTtest/visible \
--outdir ./results/ldm_kaist_s200 \
--config ./configs/latent-diffusion/kaist256-vqf4_rec_tev.yaml \
--checkpoint ./pretrained/kaist_tev/kaist_tev.ckpt