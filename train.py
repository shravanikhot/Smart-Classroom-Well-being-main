import tensorflow as tf
import os
import cv2
import numpy as np

def create_tfrecord(dataset_dir, output_file):
    # Create a writer
    writer = tf.io.TFRecordWriter(output_file)

    # Map class names to integers
    classes = sorted(os.listdir(dataset_dir))
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for class_name, label in class_to_label.items():
        class_path = os.path.join(dataset_dir, class_name)

        # Iterate through images in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image {img_path}")
                continue

            img = cv2.resize(img, (224, 224))  # Resize to a fixed size
            img_data = img.tobytes()

            # Create a feature dictionary
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }

            # Create an Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Write to the TFRecord file
            writer.write(example.SerializeToString())

    writer.close()
    print(f"TFRecord saved to {output_file}")

def parse_tfrecord(example_proto):
    # Define the feature description
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the image and label
    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image, (224, 224, 3))
    label = tf.cast(parsed_features['label'], tf.int32)
    return image, label

def train_model(tfrecord_file, num_classes, epochs=10, batch_size=32):
    # Load the dataset
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Define MobileNetV2 model
    try:
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze the base model

        # Add classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    except Exception as e:
        print(f"Error initializing the model: {e}")
        return

    # Compile the model
    try:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    except Exception as e:
        print(f"Error compiling the model: {e}")
        return

    # Train the model
    try:
        model.fit(dataset, epochs=epochs)
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save the model
    try:
        model.save("trained_mobilenetv2_model.h5")
        print("Model saved as trained_mobilenetv2_model.h5")
    except Exception as e:
        print(f"Error saving the model: {e}")
        return


if __name__ == "__main__":
    dataset_directory = "E:/Smart Classroom/dataset"  # Replace with your dataset directory
    output_tfrecord = "train.tfrecord"
    create_tfrecord(dataset_directory, output_tfrecord)

    # Number of classes in your dataset
    num_classes = len(os.listdir(dataset_directory))

    # Train the model
    train_model(output_tfrecord, num_classes)
