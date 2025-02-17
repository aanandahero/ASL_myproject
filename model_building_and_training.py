import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, \
    Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Define dataset path
dataset_dir = r"C:\Users\Aananda Sagar Thapa\OneDrive\Desktop\ASL_Alphabet_Dataset\asl_alphabet_train"

# Image properties
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 11


def inverted_residual_block(x, in_channels, out_channels, expansion_factor, stride):
    """Creates an inverted residual block (like MobileNetV2)"""

    # Expansion
    expanded_channels = in_channels * expansion_factor
    x = Conv2D(expanded_channels, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)  # ReLU6 activation

    # Depthwise Convolution
    x = DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # Projection (Linear Bottleneck)
    x = Conv2D(out_channels, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Residual Connection (if stride is 1 and input channels match output channels)
    if stride == 1 and x.shape[-1] == out_channels:
        shortcut = x  # Store the original input
        x = Add()([x, shortcut])  # Add the residual connection

    return x


def custom_mobilenetv2(input_shape=(128, 128, 3), num_classes=29):
    """Builds a custom CNN similar to MobileNetV2"""

    inputs = Input(shape=input_shape)

    # Initial Convolution Layer
    x = Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # Inverted Residual Blocks (like MobileNetV2)
    x = inverted_residual_block(x, in_channels=32, out_channels=16, expansion_factor=1, stride=1)
    x = inverted_residual_block(x, in_channels=16, out_channels=24, expansion_factor=6, stride=2)
    x = inverted_residual_block(x, in_channels=24, out_channels=24, expansion_factor=6, stride=1)
    x = inverted_residual_block(x, in_channels=24, out_channels=32, expansion_factor=6, stride=2)

    # Global Average Pooling and Output Layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True
)

# Load Data
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Get class labels
num_classes = len(train_generator.class_indices)

# Build and compile the model
model = custom_mobilenetv2(num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    verbose=1
)

# Save the trained model
model.save("asl_mobilenetv2_model.h5")
print("MobileNetV2-based ASL model saved!")

# Save the training history
with open("training_history_mobilenetv2.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Training history saved!")
