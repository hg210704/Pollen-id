# Pollen ID

Identify pollen from light-microscopy (LM) images: it returns the **genus** and the **species**,
the five closest matches, a confidence value, and an attention map (Grad-CAM). A web page shows
training progress, run comparisons, evaluation results, the dataset and model versions.

Images come from [PalDat](https://www.paldat.org). Personal study use only; do not redistribute the
images and cite PalDat in anything you present.

## What is in the box

| Folder | What it does |
|---|---|
| `pollen/` | the library: data splitting, model, training, evaluation, prediction, Grad-CAM |
| `scripts/` | commands you run (prepare data, train, evaluate, predict, export, add genera) |
| `api/` | FastAPI server (also serves the web page) |
| `web/` | the web interface (plain HTML/CSS/JS, no Node needed; Chart.js is bundled) |
| `tests/`, `tools/` | tests and a fake-data generator for trying everything quickly |
| `config.yaml` | every setting in one place |

## 1. Install (Windows or Linux)

You need Python 3.10 or newer. In a terminal inside this folder:

```
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

For a GPU, install the matching PyTorch first from https://pytorch.org/get-started/locally/,
then run the `pip install` above. Training on CPU works but is slow; a GPU with 8 GB or more is
comfortable for the default `convnext_tiny` at 384 px (lower `data.image_size` or `train.batch_size`
if memory runs out).

## 2. Put the dataset in place

Copy the folder made by the collection script so it looks like this, next to `config.yaml`:

```
paldat_images/
  metadata.csv
  LM/
    Acer_campestre/Acer_campestre_LM_1.jpg ...
    Betula_pendula/ ...
```

Somewhere else? Change `data.root` and `data.metadata_csv` in `config.yaml`.
The set of genera and species is read from the folders every time. Nothing is hard-coded to a letter range.

## 3. Try everything on fake data first (recommended, about 5 minutes)

```
python -m pytest tests -q
```

This builds a tiny fake dataset, trains for a few epochs, evaluates, predicts, and calls the API.
All tests should pass before you spend hours on real training.

## 4. Real run

```
python scripts/prepare_data.py          # checks images, removes duplicates, makes leakage-free splits
python scripts/train.py                 # trains and evaluates; takes a while
python scripts/evaluate.py --run latest --gradcam 12
python scripts/predict.py some_pollen.jpg
uvicorn api.main:app --port 8000        # then open http://localhost:8000
```

`prepare_data.py` prints a summary. **Read it.** It lists species that have too few images or specimen
records. Those are the ones that limit accuracy.

Useful options:

```
python scripts/train.py --backbone tf_efficientnetv2_s
python scripts/train.py --set data.image_size=448 --set train.epochs_finetune=60
python scripts/train.py --set train.mixup_alpha=0.2 --set train.cutmix_alpha=1.0
python scripts/compare_backbones.py --quick     # convnext_tiny vs efficientnetv2_s vs deit_small
python scripts/export_model.py --format torchscript
```

## 5. Adding more genera later

```
python scripts/add_genera.py --src D:\more\paldat_images --train finetune   # fast, starts from current model
python scripts/add_genera.py --src D:\more\paldat_images --train retrain    # slower, usually most accurate
```

Existing train/validation/test assignments are kept, so an image that was in the test set never moves
into training. Every run is saved under `runs/`; activate or roll back on the "Model versions" page.

## Synthetic data and early stopping

**Synthetic images** (`python scripts\make_synthetic.py`) are random variants of your TRAIN photos: any
rotation, stain colour, focus blur, brightness, noise, zoom. They teach the model to ignore those things and
they balance species that have only two or three photos. They are added to the training split only, so
validation and test images stay real and reported accuracy still measures real photos. If you rebuild the
splits later, variants of a photo that moved to val/test are dropped automatically. A variant carries no new
botanical information, so extra *real* images (more specimens per species) will always help more.
Settings: `data.use_synthetic`, `data.synthetic_target_per_species`, `data.synthetic_max_per_source`.

**Early stopping** watches the validation loss (`train.early_stopping_metric: val_loss`). A single epoch
where it goes up is normal noise; training stops only after `early_stopping_patience` epochs in a row without
an improvement of at least `early_stopping_min_delta`, and the model keeps the weights of the BEST epoch,
not the last one. The validation set is small when most species have only a few photos, so it is noisy;
more real images make it reliable.

Order of commands after changing the data:

```
python scripts\prepare_data.py --reset-splits
python scripts\make_synthetic.py --clean
python -m pollen.train --note "synthetic + early stopping"
```

## How accuracy is measured (and why it can be trusted)

* **No leakage.** Photos from the same specimen record (same PalDat page) look nearly identical. All
  images of a record, and any records sharing an identical image, always land in the same split.
* **Species with 1 record cannot be tested**, so they are used for training only and flagged. Species
  with 2 records get train + test; 3 or more get train + validation + test.
  (`data.image_level_split_for_single_record: true` changes this, but the numbers become optimistic.)
* **Test results are only reported, never used to choose anything.** Backbones are chosen on validation data.
* Genus is predicted first; the species is the best one *inside* that genus.
* The model can answer **"uncertain"** when confidence is below a threshold tuned on validation data,
  instead of confidently naming the wrong species.

### The 90-100 % target

The code aims for 90 % or more and checks it (`target_genus_acc`, `target_species_acc` in `config.yaml`).
Whether it is reached depends on the data: many PalDat species have only a handful of LM images and
some congeneric species look alike. `runs/<run>/report.md` states plainly whether the target was met and,
if not, which species and how much data are limiting. Do not trust any number the project does not
print from the held-out test set.

## Layout of a run (`runs/<run_id>/`)

`config.json` settings, `history.json` per-epoch curves, `status.json` live status, `model.pt` best
weights, `metrics.json` headline numbers, `evaluation.json` confusion matrices and per-species scores,
`report.md` plain-language summary, `gradcam/` attention maps of mistakes, `train.log` (runs started from the web page).

## Troubleshooting

* *"No trained model yet"*: train one (step 4) or start a run on the Training page.
* *Out of memory*: lower `train.batch_size` or `data.image_size`.
* *Pretrained weights fail to download*: they need internet on first use; or set `model.pretrained: false` (lower accuracy).
* *Port in use*: change `--port 8000`.
