# only works if this is the first library imported on my system??
# straight up the weirdest thing I've ever seen, python sucks
from sentence_transformers import SentenceTransformer

from flask import Flask, request
from flask_restful import Resource, Api

from tensorflow import keras
import numpy as np


app = Flask(__name__)
api = Api(app)

# embedding model
emb_model=SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# generating model
model_path = './saved_model'
model = keras.models.load_model(model_path)

def gen_example(prompt):
    embedding = emb_model.encode(prompt).reshape(1,384)
    noise = np.random.normal(0, 1, 20).reshape(1,20)
    predictions = model.predict([embedding, noise], verbose=False) 
    # Convert softmax output to int representation
    max_indices = np.argmax(predictions, axis=-1)
    return max_indices[0]

class Test(Resource):
	def get(self):
		return {"response": "Hello World!"}

class Generate(Resource):
	def post(self):
		data = request.get_json()
		if data:
			if 'prompt' in data:
				prompt = data['prompt']
				return gen_example(prompt).tolist()
		else:
			return {'error': 'No description was provided in the POST request.'}, 400

api.add_resource(Generate, '/generate')
api.add_resource(Test, '/')


if __name__ == '__main__':
    app.run(debug = True)