TS2Vec on local UCR data
========================

``examples/ts2vec_ucr.py`` is a compact, runnable fixed-length UCR workflow. It
reads two local tab-separated files (one series per row, class label in the
first column, remaining columns are timesteps), trains TS2Vec, then evaluates
full-series representations with the reference RBF SVM protocol. No data
download is performed. The test labels are used only for final accuracy;
classifier fitting and any cross-validation use training embeddings and labels.

.. code-block:: bash

   python examples/ts2vec_ucr.py /data/UCR/ECG200/ECG200_TRAIN.tsv \
       /data/UCR/ECG200/ECG200_TEST.tsv --seed 0 --steps 200

The script accepts ``--batch-size``, ``--output-dims``, ``--hidden-dims``,
``--depth``, and ``--checkpoint`` as well. Defaults use a small fixed 200-step
training budget and AdamW (learning rate 0.001, weight decay 0.01). Output is
one JSON record with accuracy, effective settings, and library versions. This
recipe is intended to demonstrate the API, not to claim a benchmark result.

Input batches have the native dictionary form ``{"series": [B,T,1],
"label": ...}``. At evaluation, ``model.eval(); model(batch, stage="test")``
produces a timestamp representation for each point and a full-series embedding
formed by temporal max pooling. Evaluation uses the cumulative parameter average
of the initial encoder and completed optimizer updates. The recipe does not
select a checkpoint on the test set.

This example covers timestamped fixed-length classification embeddings only.
It does not implement forecasting, anomaly detection, sliding-window
embeddings, or multiscale representation extraction.

Reference comparison
--------------------

Use the authors' implementation at commit
``b0088e14a99706c05451316dc6db8d3da9351163``. With its dependencies installed
and UCR data under ``datasets/UCR/``, run:

.. code-block:: bash

   git clone https://github.com/zhihanyue/ts2vec.git
   cd ts2vec
   git checkout b0088e14a99706c05451316dc6db8d3da9351163
   python train.py ECG200 matched --loader UCR --seed 0 --iters 200 \
       --batch-size 16 --repr-dims 320 --eval

The reference runs its ``tasks/classification.py`` full-series encoder and
SVM protocol (including training-only cross-validation when the class counts
allow it). This example mirrors that parameter search and does not scale
features. To compare fairly, use identical UCR rows, seed, iteration budget,
batch size, and dimensions. No matched benchmark scores are claimed by this
example. Record the reference commit and commands with any future comparison. The local recipe's
AdamW weight decay (0.01) is also explicit; match the baseline optimizer
settings before attributing score differences to implementation.

For classes with fewer than five training examples, this recipe skips
cross-validation and uses the reference small-data SVM setting. Grid search
runs serially to keep the example usable on small machines. The optional
``--checkpoint`` saves a model state dictionary, including both encoders and
the averaging counter; use Lightning checkpoints to resume optimizer state.
