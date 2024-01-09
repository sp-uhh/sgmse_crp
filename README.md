# Single and Few-step Diffusion for Generative Speech Enhancement

This repository contains the official PyTorch implementations for the 2023 paper:

- *Single and Few-step Diffusion for Generative Speech Enhancement* [1]

This repository builds upon our previous work: https://github.com/sp-uhh/sgmse and https://github.com/sp-uhh/sgmse-bbed.
Find audio examples here https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/crp.


## Installation
- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.



## Training
Training is done by executing `train_resume.py` and requires a checkpoint trained from https://github.com/sp-uhh/sgmse or https://github.com/sp-uhh/sgmse-bbed. Checkpoints are provided (see further below). A running example can be run with

```bash
python train_resume.py --base_dir <your_base_dir> --N_min 1 --N_max 1 --t_rsp_min 0.5 --pre_ckpt <path_to_ckpt/name_ckpt.ckpt> --batch_size 16 --t_rsp_max 0.5 --nolog --lr 0.00001 --loss_type default 
--stop_iteration_random last --inference_N 1 --inference_start 0.5
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files. 'path_to_ckpt/name_ckpt.ckpt' is the path to the pretrained checkpoint. We advise to set N_min, N_max, --inference_N to be the same number, also inference_start and t_rsp_max should be the same number. Moreover, stop_iteration_random has to be set to last. This command with the correct checkpoint reproduces the result from [1] for training CRP with 1 step starting the reverse process from 0.5.

To get the training set "wsj0 chime3", we refer to [https://github.com/sp-uhh/sgmse](https://github.com/sp-uhh/sgmse/tree/main/preprocessing) and execute create_wsj0_chime3.py.


## Checkpoints
Pretrained checkpoints from the first training stage:
wsj0-chime3: Download https://drive.google.com/file/d/1_h7pH6o-j7GV_E69SbRQF2BMRlC8tmz_/view?usp=share_link. This is the checkpoint that was used to produce the results in https://github.com/sp-uhh/sgmse-bbed for the wsj0-chime3 dataset.

voicebank-demand: Download https://drive.google.com/file/d/1AJmEJalqJyrgZEVh-NZ2mgHdIeu-XgMz/view?usp=drive_link. This is the checkpoint where BBED was trained on VoiceBank-Demand.

Checkpoints from the second training stage:
https://drive.google.com/file/d/1E0-Cr5CX7xNr_T53eVZP-1-dvlmBAJW6/view?usp=drive_link. This is the checkpoint when we load the above checkpoint (https://drive.google.com/file/d/1_h7pH6o-j7GV_E69SbRQF2BMRlC8tmz_/view?usp=share_link) and run train_resume.py with the command from above. It therefore reproduces the result from [1] (CRP one step on wsj0-chime3 with BBED).



To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.

## Evaluation

To evaluate on a test set, run
```bash
python eval.py --test_dir <your_test_dir> --type <your_enhanced_dir> --ckpt <path_to_model_checkpoint>  --N 1 --reverse_starting_point 0.5
```

starts enhancement from 0.5 with 1 reverse step. --ckpt must be now the trained model after the second training stage, e.g. the checkpoint https://drive.google.com/file/d/1E0-Cr5CX7xNr_T53eVZP-1-dvlmBAJW6/view?usp=drive_link from above.




## Citations / References

We kindly ask you to cite our paper (can be found on https://arxiv.org/abs/2309.09677) in your publication when using any of our research or code:

>[1] Bunlong Lay, Jean-Marie Lermercier, Julius Richter and Timo Gerkmann. *Single and Few-step Diffusion for Generative Speech Enhancement*, ICASSP, 2024.
