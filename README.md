# Speech Enhancement and Dereverberation with Diffusion-based Generative Models

This repository contains the official PyTorch implementations for the 2022 papers:

- *Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain* [1]
- *Speech Enhancement and Dereverberation with Diffusion-Based Generative Models* [2]


## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.


## Pretrained checkpoints

We provide pretrained checkpoints for the models trained on VoiceBank-DEMAND and WSJ0-CHiME3, as in the paper. They can be downloaded at https://drive.google.com/drive/folders/1CSnkhUSoiv3RG0xg7WEcVapyLuwDaLbe?usp=sharing.

- For resuming training, you can use the `--resume_from_checkpoint` option of `train.py`.
- For evaluating these checkpoints, use the `--ckpt` option of `enhancement.py` (see section **Evaluation** below).


## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [2]) can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.

**Note:**
- Our Interspeech paper [1] uses `--backbone dcunet`. You need to pass `--n_fft 512` to make it work.
- Our journal preprint [2] uses `--backbone ncsnpp`.


## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
```

to generate the enhanced .wav files, and subsequently run

```bash
python calc_metrics.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir>
```

to calculate and output the instrumental metrics.

Both scripts should receive the same `--test_dir` and `--enhanced_dir` parameters. The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.


## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:

>[1] Simon Welker, Julius Richter and Timo Gerkmann. *Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain*, ISCA Interspeech, 2022.
>
>[2] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay and Timo Gerkmann. *Speech Enhancement and Dereverberation with Diffusion-Based Generative Models*, TBD, 2022.