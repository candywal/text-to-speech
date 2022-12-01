import random
import logging
import numpy as np
import tensorflow as tf
import IPython.display as ipd
import torch
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset
from transformers import TFWav2Vec2Model, AutoFeatureExtractor

BATCH = 32         # batch size for training and evaluating our model.
NUM_CLASSES = 10   # number of classes our dataset will have (11 in our case).
HIDDEN_DIM = 768   # dim of our model output (768 in our case)
SEQ_LENGTH = 16000 # length of audio
MAX_FRAMES = 49    # wav2Vec 2.0 results

MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # pretrained model

tf.get_logger().setLevel(logging.ERROR)
tf.keras.utils.set_random_seed(42)

def preprocess_function(data):
    # A feature extractor is in charge of preparing input features for audio or vision models. 
    # This includes feature extraction from sequences, e.g., pre-processing audio files to Log-Mel Spectrogram features
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_CHECKPOINT, return_attention_mask=True
    )
    audio_arrays = [x["array"] for x in data["audio"]]
    inputs = feature_extractor(
        # audio input
        audio_arrays,
        # the sampling rate at which the audio files should be digitalized expressed in Hertz per second.
        sampling_rate=feature_extractor.sampling_rate,
        # maximum length of the returned list and optionally padding length
        max_length=SEQ_LENGTH,
        # activates truncation to cut input sequences longer than max_length to max_length.
        truncation=True,
        # pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
        padding=True,
    )
    return inputs

def mean_pool(hidden_states, feature_lengths):
    attenion_mask = tf.sequence_mask(
        feature_lengths, maxlen=MAX_FRAMES, dtype=tf.dtypes.int64
    )
    padding_mask = tf.cast(
        tf.reverse(tf.cumsum(tf.reverse(attenion_mask, [-1]), -1), [-1]),
        dtype=tf.dtypes.bool,
    )
    hidden_states = tf.where(
        tf.broadcast_to(
            tf.expand_dims(~padding_mask, -1), (BATCH, MAX_FRAMES, HIDDEN_DIM)
        ),
        0.0,
        hidden_states,
    )
    pooled_state = tf.math.reduce_sum(hidden_states, axis=1) / tf.reshape(
        tf.math.reduce_sum(tf.cast(padding_mask, dtype=tf.dtypes.float32), axis=1),
        [-1, 1],
    )
    return pooled_state

# combines encoder/decoder for an E2E model
class TFWav2Vec2ForAudioClassification(layers.Layer):
    def __init__(self, model_checkpoint, num_classes):
        super(TFWav2Vec2ForAudioClassification, self).__init__()
        # init wav2vec 2.0 model without the classification head
        self.wav2vec2 = TFWav2Vec2Model.from_pretrained(
            model_checkpoint, 
            apply_spec_augment=False, 
            from_pt=True
        )
        self.pooling = layers.GlobalAveragePooling1D()
        # drop-out layer before the final classification head
        self.intermediate_layer_dropout = layers.Dropout(0.5)
        # classification head with softmax activation
        self.final_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # use first output in the returned dictionary corresponding to output of the last layer
        hidden_states = self.wav2vec2(inputs["input_values"])[0]

        # if attention mask does exist then mean-pool only un-masked output frames
        if tf.is_tensor(inputs["attention_mask"]):
            # length of each audio input by summing up the attention_mask
            # (attention_mask = (BATCH x SEQ_LENGTH) ∈ {1,0})
            audio_lengths = tf.cumsum(inputs["attention_mask"], -1)[:, -1]
            # retrive numbe of wav2Vec output frams for each inputted audio length
            feature_lengths = self.wav2vec2.wav2vec2._get_feat_extract_output_lengths(
                audio_lengths
            )
            pooled_state = mean_pool(hidden_states, feature_lengths)
        # if attention mask does not exist then mean-pool only all output frames
        else:
            pooled_state = self.pooling(hidden_states)

        intermediate_state = self.intermediate_layer_dropout(pooled_state)
        final_state = self.final_layer(intermediate_state)

        return final_state

def build_model():
    inputs = {
        "input_values": tf.keras.Input(shape=(SEQ_LENGTH,), dtype="float32"),
        "attention_mask": tf.keras.Input(shape=(SEQ_LENGTH,), dtype="int32"),
    }
    # init wav2vec from facebook, include filtered classes
    wav2vec2_model = TFWav2Vec2ForAudioClassification(MODEL_CHECKPOINT, NUM_CLASSES+2)(inputs)

    model = tf.keras.Model(inputs, wav2vec2_model)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model

def main():
    # load model, ignore checksum
    init_dataset = load_dataset("superb", "ks", ignore_verifications=True)
    print("init dataset:\n", init_dataset)

    # filter 1/2 from existing training set for both train and test data
    init_dataset = init_dataset["train"].train_test_split(
        train_size=0.5, test_size=0.5, stratify_by_column="label"
    ).filter(
        lambda x: x["label"]
        != (
            init_dataset["train"].features["label"].names.index("_unknown_")
            and init_dataset["train"].features["label"].names.index("_silence_")
        )
    )

    # batch training/testing data
    init_dataset["train"] = init_dataset["train"].select(
        [i for i in range((len(init_dataset["train"]) // BATCH) * BATCH)]
    )
    init_dataset["test"] = init_dataset["test"].select(
        [i for i in range((len(init_dataset["test"]) // BATCH) * BATCH)]
    )
    print("dataset:\n", init_dataset)
    labels = init_dataset["train"].features["label"].names

    # pre-process our init_dataset dataset, removed unused columns
    processed_init_dataset = init_dataset.map(
        preprocess_function, remove_columns=["audio", "file"], batched=True
    )

    # load dataset as dict numpy arrays
    train = processed_init_dataset["train"].shuffle(seed=42).with_format("numpy")[:]
    test = processed_init_dataset["test"].shuffle(seed=42).with_format("numpy")[:]

    model = build_model()

    train_x = {x: y for x, y in train.items() if x != "label"}
    test_x = {x: y for x, y in test.items() if x != "label"}

    model.fit(
        train_x,
        train["label"],
        validation_data=(test_x, test["label"]),
        batch_size=BATCH,
        epochs=1,
    )
    model.save_model("./text-to-speech-TFWav2Vec2Model")

if __name__ == "__main__":
    main()
