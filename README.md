This repo includes a single model (no ensemble) for [toxic classification challange](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/overview). The public LB score is ~ 0.93.
The backbone model from Hugginface is finetuned in the notebook _pytorch_model.ipynb_.
The model trained using Google Cloud: https://www.kaggle.com/quanncore/xlm-roberta-large
The notebook for model inference on test data of toxic classification: https://www.kaggle.com/quanncore/pytorch-tpu-inference

Several methods applied in the model:
- Translated training data (from English to other languages) have been used.
- Further fine-tuning has been performed on the validation dataset.
- Data augmentation on the training set has been used.
- Class balancing on a distributed environment has been used.
- TPU is supported.
- A multisample dropout network has been used on the classifier head.
- The hidden states of transformers have been aggregated by attention weights instead of using the last layer.
- Different learning rates for the backbone model and classifier head have been used.
- Stochastic Weight Averaging (SWA) has been used.

Reproduction step of training:
- Create a _credential_ folder and put the Github and Kaggle credential files in JSON format.
- Create a _data_ folder and put the toxic dataset on it. You can change the data paths on the training notebook.
- Create a _model_ folder for storing different training model versions.
- Create a _output_ folder for storing submission files if needed.
