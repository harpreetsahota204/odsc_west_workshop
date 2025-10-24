# Mastering Visual AI with Vision-Language Models Workshop at ODSC West Oct 2025 in SF


# What You Need for this Workshop

First, clone this repository:

`git clone https://github.com/harpreetsahota204/odsc_west_workshop`

Then, you'll need to set up a virtual enviornment with everything outlined in [requirements.txt](requirements.txt).

Which you can do by running the following:

`pip install -r requirements.txt`

## Free Disk Space

The datasets we are downloading are fairly large. You'll want to ensure you have at least 10GB of free disk space. 

After the workshop, you can find these files in the following locations to free up disk space:

- **macOS**: `~/fiftyone/huggingface/hub/`
- **Linux**: `~/fiftyone/huggingface/hub/`
- **Windows**: `C:\Users\<YourUsername>\fiftyone\huggingface\hub\`

You can safely delete the entire `fiftyone` directory if you no longer need the cached datasets.

## Sign Up for a Hugging Face Account

Since the datasets we're downloading are fairly large you may experience some rate limiting from Hugging Face. 

It's a good idea to sign up for a Hugging Face account and get a token, though not required.

Follow the instructions [here](https://huggingface.co/docs/hub/en/security-tokens) to learn how to get your token. Once you have that token, sign in to Hugging Face using your terminal by entering:

`hf auth login`

## Will I Need Any Other API Keys?

No. Everything we are doing here is completely open source, `pip` installable, and works on your local host...unless you're planning on running these notebooks on Google Colab, then you'd need a Google account.

## Will I need GPU access for this workshop?

For the purposes of this workshop I've uploaded the datasets with all the enrichments made using VLMs to Hugging Face. Instructions for downloading the datasets are shown later in this document.

You'll need access to a GPU runtime if you plan on running this code **after** the workshop.

#  Installing FiftyOne

The full installation guide, plus troubleshooting can be found [here.](https://docs.voxel51.com/getting_started/install.html)

## Step 1: Install Miniconda or Anaconda
Conda is the recommended environment manager. If you don't already have Conda installed:

1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for smaller footprint) or [Anaconda](https://www.anaconda.com/products/distribution)

2. Run the installer and follow the prompts

## Step 2: Create a Conda environment

Open your terminal or Anaconda Prompt and run:
```bash
conda create -n fiftyone python=3.11
```

This creates a new environment named "fiftyone" with Python 3.11 (compatible with FiftyOne).

## Step 3: Activate the environment

```bash
conda activate fiftyone
```
Your prompt should change to indicate the environment is active.

## Step 4: Install FiftyOne

Install FiftyOne using pip within the Conda environment:
```bash
pip install fiftyone
```

## Step 5: Verify installation

```bash
python -c "import fiftyone as fo; print(fo.__version__)"
```

This should print the FiftyOne version without errors.

## Step 6: Install dependencies

You can install the required dependencies for this workshop using the [`requirements.txt`](requirements.txt) in this repository

```bash
pip install -r requirements.txt
```

## Step 7: Quick test

Run a simple test to ensure everything works:
```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load a small sample dataset
dataset = foz.load_zoo_dataset("quickstart")

# Launch the app to explore it
session = fo.launch_app(dataset)
```

## Working with your FiftyOne installation

- To use FiftyOne later, always activate the environment first: `conda activate fiftyone`
- To upgrade FiftyOne: `pip install --upgrade fiftyone`
- To deactivate when finished: `conda deactivate`

## Troubleshooting

- If installation fails, try: `pip install --upgrade pip setuptools wheel build`

- For platform-specific issues, refer to the [FiftyOne documentation](https://beta-docs.voxel51.com/getting_started/basic/install/#troubleshooting)

- Mac users may need XCode Command Line Tools

- Linux users might need additional packages like `python3-dev`

This conda-based installation creates an isolated environment that won't interfere with your other Python projects, making it easy to manage dependencies and keep your FiftyOne setup clean.

You can [read these docs for more detail](https://beta-docs.voxel51.com/fiftyone_concepts/running_environments/) about setting up your enviornment.

# Downloading the Datasets for this Workshop

It's best that you download the required dataset prior to attending the workshop to avoid any rate limits from Hugging Face or slow internet connection speeds at the venue.

We'll make use of the [CarDD dataset](https://huggingface.co/datasets/harpreetsahota/CarDD).

You can download this dataset as follows:

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "harpreetsahota/CarDD",
    name="cardd_from_hub",
    # max_samples=500, # if you want to work with a subset of the dataset
    persistent=True,
    overwrite=True,
    )
```

Throughout the workshop, we'll enrich this dataset using various vision language models. Since these models are quite large and require GPUs to run it won't be feasible to run them live in the session.

I've already parsed the dataset in it's "final state" and have uploaded that to Hugging Face as well. You can download that as follows:

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "harpreetsahota/cardd_workshop_post_03",
    overwrite=True,
    persistent=True
    )
```

There is also a test split of the CarDD dataset which we will also make use of. You can download that as follows:

```python
from fiftyone.utils.huggingface import load_from_hub

test_dataset = load_from_hub(
    "harpreetsahota/cardd_test_post_03",
    overwrite=True,
    persistent=True
    )
```

# Install FiftyOne Plugins

You'll also want to install the following plugins for FiftyOne, as we will be making use of them during the workshop.

---
The [Dashboard Plugin](https://docs.voxel51.com/plugins/plugins_ecosystem/dashboard.html), which can be installed by running the following in your terminal:

```bash
fiftyone plugins download \
    https://github.com/voxel51/fiftyone-plugins \
    --plugin-names @voxel51/dashboard
```
---
The [Model Evaluation Panel](https://docs.voxel51.com/user_guide/app.html#app-model-evaluation-panel), which can be installed by running the following in your terminal:

```bash
fiftyone plugins download \
    https://github.com/voxel51/fiftyone-plugins \
    --plugin-names @voxel51/evaluation
```
---
The [Caption Viewer plugin](https://github.com/harpreetsahota204/caption_viewer), which can be installed by running the following in your terminal:

```bash
# Install from GitHub
fiftyone plugins download https://github.com/harpreetsahota204/caption-viewer
```

# Download a Fine-Tuned Model

A part of this workshop discussed model evalution techniques. For this, we need a fine-tuned model. I've already trained the model and have [uploaded the weights to Hugging Face](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11).

You can download these model weights by simply [clicking this link](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11/resolve/main/best.pt?download=true) and then moving the checkpoint into this directory and renaming it to `yolov11-seg-cardd.pt`

Or, you can download programmatically using something like `wget`:

```bash
wget https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11/resolve/main/best.pt -O yolov11-seg-cardd.pt
```

Or just using Python:

```python
import urllib.request

url = "https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11/resolve/main/best.pt"
output_path = "yolov11-seg-cardd.pt"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, output_path)
print(f"Downloaded to {output_path}")
```


# [Optional] Additional Materials and References

This workshop is an abbreviated version of a much longer workshop that I host. I encourage you to check out the notebooks for that workshop if you're interested in learning more. You can find these materials [here](https://github.com/harpreetsahota204/car_dd_dataset_workshop).

If you're interested in a deeper dive into data quality and it's impact on object detection use cases, you can freely audit this course I created on Coursera: [Hands-on Data Centric Visual AI](https://www.coursera.org/learn/hands-on-data-centric-visual-ai/).

