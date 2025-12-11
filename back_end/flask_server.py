from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from predict import predict
import base64

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

#localhost/5000
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status" : "Server is running on 5000"})

#localhost/5000/test
#or
#localhost/5000/test?name=Earth outputs Hello, Earth
@app.route("/test", methods=["GET"])
def test():
    name = request.args.get("name", "World")
    return jsonify({"message": f"Hello, {name}"})



@app.route("/sendImage", methods=["POST", "OPTIONS"])
@cross_origin()
def sendImage():

    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    data = request.get_json()

    if not data:
        return jsonify({"error" : "Could not get data"}), 400

    bytes = base64.b64decode(data.get('fileData'))
    prediction = predict(bytes)
    return jsonify({"prediction" : prediction}), 200

    fileName = data.get('fileName')
    return jsonify({"message" : f"got it : {fileName}"}), 200


def _build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight successful'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response



def not_found(e):
    return jsonify({"error" : "Not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    print("hello")