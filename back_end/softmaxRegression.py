from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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


def not_found(e):
    return jsonify({"error" : "Not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    print("hello")