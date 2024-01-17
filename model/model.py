import keras
from keras import layers


def lstm():
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(
    400, 6)))  # , return_sequences=True)) # considering the length of the arrays is going to be the same

    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(64, activation='relu'))

    # softmax
    model.add(keras.layers.Dense(7, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.000001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


#
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer(
        input_shape=(400, 6),
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        dropout=0,
        mlp_dropout=0,
):
    n_classes = 7

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=0.00005)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def save_model(model, path="saved_models/model.keras"):
    keras.saving.save_model(model, path, overwrite=True)


def load_model(path="saved_models/model.keras"):
    return keras.saving.load_model(path)
