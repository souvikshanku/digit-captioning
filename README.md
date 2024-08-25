# digit-captioning
<!-- markdownlint-disable MD033 MD045 -->

<p align="center">
  <img src=lstm.png height="300"/>
</p>

Digit captioning using CNN and LSTM, broadly based on the image captioning model described in [this paper](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf) by Karpathy et al.

## Example Usage

```bash
git clone https://github.com/souvikshanku/digit-captioning.git
cd digit-captioning

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the app
python app.py
```

The model has been trained on double digit MNIST dataset obtained from [here](https://github.com/shaohua0116/MultiDigitMNIST). To train the model from scratch download the dataset and see `repro.ipynb`.

```bash
# Download, unzip and move
cd digit-captioning/data
gdown https://drive.google.com/uc?id=1NMLh34zDjrI-bOIK6jgLJAqRrUY3uETC
unzip double_mnist.zip
mv labels.csv data/labels.csv
```
