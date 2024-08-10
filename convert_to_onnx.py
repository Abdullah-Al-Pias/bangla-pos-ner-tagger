import tensorflow as tf
import tf2onnx

def convert_model_to_onnx(h5_model_path, onnx_model_path):
    # Load the trained Keras model
    model = tf.keras.models.load_model(h5_model_path)

    # Define the input signature for the model
    input_signature = (tf.TensorSpec((None, None), tf.float32, name="input"),)

    # Convert the model to ONNX format
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    # Save the ONNX model to a file
    with open(onnx_model_path, "wb") as f:
        f.write(model_proto.SerializeToString())

if __name__ == "__main__":
    h5_model_path = 'bangla_pos_ner_model.h5'  # Path to your Keras model
    onnx_model_path = 'bangla_pos_ner_model.onnx'  # Desired path for the ONNX model

    convert_model_to_onnx(h5_model_path, onnx_model_path)
    print(f"Model successfully converted to ONNX format and saved at: {onnx_model_path}")
