
Run command to train the model:

``python train.py --batch_size 32 --update_freq 1 --lr 0.01 --epochs 1000 --eval_every 2 --check_every 20 --diffusion_steps 1000 --transformer_depth 12 --transformer_heads 16 --transformer_local_heads 8 --gamma 0.99 --log_wandb False``


Run command to sample new trajectories based on the trained model (the checkpoint is saved in "~/log/flow/bandit/multinomial_diffusion_v2/expdecay/YYYY-MM-DD_hh-nn-ss"):

``python eval_sample.py --length 512 --model "~/log/flow/bandit/multinomial_diffusion_v2/expdecay/YYYY-MM-DD_hh-nn-ss" --samples 16``

The length should match the shape of samples (The (state-)action sequence length). Modify the file "/datasets/data.py" and "/datasets/dataset_bandit.py" to align with your data.
