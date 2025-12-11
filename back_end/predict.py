import os
from back_end.train_model import load_model_for_inference, train_dataset, predict_image

if __name__ == "__main__":
    # example usage on a new image
    image_path = "test_imgs/toyota_camry.jpg"  # set your path here
    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist")
    else:
        # load model and predict
        inference_model = load_model_for_inference("car_classifier_weights.pth")
        class_names = train_dataset.classes  # e.g. ['sedan','SUV','truck']

        pred_class, pred_idx = predict_image(image_path, inference_model, class_names)
        print(f"Predicted class: {pred_class} (index {pred_idx})")