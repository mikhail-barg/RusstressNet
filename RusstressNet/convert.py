from keras.models import model_from_json
import keras2onnx
import onnx
import onnxruntime

with open('/home/user/russtress/dev/russtress/russtress/text_model.json', 'r') as content_file:
	json_string = content_file.read()
model = model_from_json(json_string)
model.load_weights('/home/user/russtress/dev/russtress/russtress/weights.96.hdf5')

onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, '/home/user/russtress/dev/model.onnx')

sess = onnxruntime.InferenceSession( '/home/user/russtress/dev/model.onnx')