# --- Grad-CAM helper class ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # forward & backward hooks (full backward hook to avoid warning)
        self.handle_fwd = target_layer.register_forward_hook(self._forward_hook)
        self.handle_bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # keep on same device as module output
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output\[0] is the gradient wrt module output
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: shape \[1, 3, H, W], on same device as model
        """
        device = input_tensor.device

        # forward
        output = self.model(input_tensor)  # \[1, num_classes]
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # backward w.r.t. target_class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # activations: \[1, C, H', W'], gradients: \[1, C, H', W']
        grads = self.gradients          # stays on device
        acts = self.activations         # stays on device

        # global-average-pool gradients -> weights: \[C]
        weights = grads.mean(dim=(2, 3))[0]  # \[C]

        # weighted sum of activations; keep cam on same device
        cam = torch.zeros(acts.shape[2:], dtype=torch.float32, device=device)
        for c, w in enumerate(weights):
            cam += w * acts[0, c, :, :]

        # ReLU and normalize
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()  # move to cpu only at the end

    def close(self):
        self.handle_fwd.remove()
        self.handle_bwd.remove()


# --- utility: denormalize and overlay heatmap ---
def overlay_cam_on_image(img_tensor, cam, alpha=0.5):
    """
    img_tensor: [3, H, W], normalized (ImageNet mean/std)
    cam: [H', W'] numpy array in [0,1]
    returns: RGB uint8 image with CAM overlay
    """
    # denormalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = img_tensor.cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # to HxWx3
    img = np.transpose(img, (1, 2, 0))

    # resize CAM to image size
    H, W, _ = img.shape
    cam_resized = cv2.resize(cam, (W, H))
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)

    # colormap on CAM
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # overlay
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    return overlay


# --- example usage on one validation image ---
# pick one sample from val_dataset
for i in range(len(val_dataset)):
    sample_img, sample_label = val_dataset[i]      # sample_img: PIL-transformed tensor (3x224x224)
    input_tensor = sample_img.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Grad-CAM on ResNet18 last conv layer (layer4[-1])
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # generate CAM for predicted class (or set target_class=int)
    cam = grad_cam.generate(input_tensor, target_class=None)

    # overlay heatmap on image
    overlay = overlay_cam_on_image(sample_img, cam, alpha=0.5)

    # show
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM overlay (true label: {train_dataset.classes[sample_label]})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    grad_cam.close()