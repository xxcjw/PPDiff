# PPDiff: Preference-Guided Personalized Diffusion for Item Presentation in Image-based Recommendation



## Project Structure

```
.
├── config/
│   └── Beauty/  Sports/  Toys/     # one config set per dataset
│       ├── beauty.yaml / sports.yaml / toys.yaml   # data preprocessing
│       ├── profile.yaml                            # user profile extraction
│       ├── diffusion.yaml                          # diffusion training / sampling
│       └── baseline.yaml                           # baseline evaluation
├── src/
│   ├── main.py                     
│   ├── data/                       # data preprocessing pipeline
│   ├── mask/                  
│   ├── model/
│   │   ├── latent_diffusion.py     # latent diffusion
│   │   ├── trainer.py              # trainer
│   ├── baselines/                  # image-based baselines
│   └── user_profile/              
├── pic/                            # item images and masks
└── outputs/                        # generation results
```



## Quick Start

```bash
# 1. Data preprocessing
python src/main.py preprocess --config config/Beauty/beauty.yaml

# 2. Mask extraction
python src/main.py mask --image-dir pic/Beauty --mask-dir pic/Beauty_masks \
    --item-json data/processed/Beauty/train.json data/processed/Beauty/test.json

# 3. User profile extraction
python src/main.py profile --config config/Beauty/profile.yaml

# 4. Train the generative model 
python src/main.py train --config config/Beauty/diffusion.yaml \
    --image-dir pic/Beauty --train-json data/processed/Beauty/train.json \
    --test-json data/processed/Beauty/test.json \
    --profiles-dir data/processed/Beauty/profiles --mask-dir pic/Beauty_masks

# 5. Generate visual presentation conditions for eval users' test targets
python src/main.py gen-eval-styles --config config/Beauty/profile.yaml
```

Switch datasets by replacing `config/Beauty/*.yaml` with `config/{Sports,Toys}/*.yaml`.


## Environment

```bash
pip install torch torchvision diffusers transformers accelerate pillow \
    pyyaml pandas tqdm requests numpy scipy rembg
pip install git+https://github.com/openai/CLIP.git
```

