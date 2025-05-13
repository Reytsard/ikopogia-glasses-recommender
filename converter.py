import tensorflow as tf
import tf2onnx

# Load the Sequential Keras model
model = tf.keras.models.load_model("last_model1.keras")

# Wrap it in a Functional model if it's Sequential
if isinstance(model, tf.keras.Sequential):
    inputs = tf.keras.Input(shape=(86, 86, 3), name="input")
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Define input signature
input_signature = [tf.TensorSpec([None, 86, 86, 3], tf.float32, name="input")]

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

# Save the ONNX model
with open("keras files/head_shape_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

