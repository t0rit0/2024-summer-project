This project is based on the codes of the following paper. From the repo : https://github.com/thestephencasper/latent_adversarial_training
BibTeX:
```
@article{casper2024defending,
  title={Defending Against Unforeseen Failure Modes with Latent Adversarial Training},
  author={Casper, Stephen and Schulze, Lennart and Patel, Oam and Hadfield-Menell, Dylan},
  journal={arXiv preprint arXiv:2403.05030},
  year={2024}
}
```


![fig1](lat_fig1.png)

## Setup

First, clone and navigate to the repo.

```
mkdir models
mkdir results
pip install -r requirements.txt
```

Then paste in your HF token to download Llama-2 in ```lat.py``` in the line saying ```TOKEN=''```

## Use

### Finetune initial model

Finetune Llama-2-7b-chat on 20k examples from the [Anthropic-HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset (a mixture of both preferred and rejected responses) with some examples poisoned to insert trojans. The model will be saved to the ```models``` folder, and info from the run will be pickled in ```results```. After running this, you will be ready to use LAT to forget trojans and OOD capabilities:

```python lat.py --epochs=2 --run_id=initial --save=True```

### Finetune with AT/LAT

There are a variety of ways to perform AT/LAT.

Finetune the model with latent space L2-norm adversarial perturbations to the queries at the 4th layer with a norm bound of 8 on the past-tense dataset:

```python lat.py --checkpoint=initial --forget=True --perturb_layer=4 --epsilon=8 --perturb_target=queries --run_id=lat_layer4_eps8_queries --save=True --test=past-tense```




