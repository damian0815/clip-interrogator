# clip-interrogator

The CLIP Interrogator uses the OpenAI CLIP models to test a given image against a variety of artists, mediums, and styles to study how the different models see the content of the image. It also combines the results with BLIP caption to suggest a text prompt to create more images similar to what was given.

# beam search crash demo on MPS/M1

Repro for transformers crash on MPS

## Requirements
```commandline
pip install ftfy regex tqdm transformers==4.15.0 timm==0.4.12 fairscale==0.4.4
git clone https://github.com/salesforce/BLIP
```

## Run

```commandline
PYTHONPATH=BLIP python demo_beam_search_crash.py
```
