import numpy as np
import keras
import keras.ops as ops

def find_layer_recursive(layer, name):
    """Recursively search for a layer by name in a model or layer."""
    if hasattr(layer, "name") and layer.name == name:
        return layer
    if hasattr(layer, "layers"):
        for sub_layer in layer.layers:
            found = find_layer_recursive(sub_layer, name)
            if found:
                return found
    return None

def find_last_conv_recursive(layer):
    """Recursively find the last Conv2D layer in a model or layer."""
    last_conv = None
    if isinstance(layer, keras.layers.Conv2D):
        last_conv = layer
    if hasattr(layer, "layers"):
        for sub_layer in layer.layers:
            found = find_last_conv_recursive(sub_layer)
            if found:
                last_conv = found
    return last_conv

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """Generates Grad-CAM heatmaps for model explainability."""
    
    # 1. Truly recursive layer lookup to find the target layer
    target_layer = find_layer_recursive(model, last_conv_layer_name)

    if not target_layer:
        # Fallback: Find the last Conv2D layer automatically
        target_layer = find_last_conv_recursive(model)

    if not target_layer:
        raise ValueError(f"Could not find any Conv2D layer in the model.")

    # 2. Compute Gradients based on backend
    backend = keras.backend.backend()
    
    if backend == "tensorflow":
        import tensorflow as tf
        
        # Check if the target layer is inside a nested base model
        inner_model = None
        if hasattr(model, "layers"):
            for layer in model.layers:
                if isinstance(layer, keras.models.Model):
                    if find_layer_recursive(layer, target_layer.name):
                        inner_model = layer
                        break
        
        if inner_model:
            # Create a gradient model for just the inner base model
            grad_model_inner = keras.models.Model(inner_model.inputs, [target_layer.output, inner_model.output])
            with tf.GradientTape() as tape:
                x = ops.convert_to_tensor(img_array)
                conv_outputs = None
                for layer in model.layers:
                    if layer == inner_model:
                        conv_outputs, x = grad_model_inner(x)
                    else:
                        x = layer(x)
                predictions = x
                class_idx = ops.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            grads = tape.gradient(class_channel, conv_outputs)
        else:
            # For TF, we still use the Functional Model approach as it's more stable there
            # but we ensure the model is functionalized if needed
            if not (hasattr(model, "input") and hasattr(model, "output")):
                inputs = keras.layers.Input(shape=img_array.shape[1:])
                outputs = model(inputs)
                func_model = keras.models.Model(inputs, outputs)
                target_layer = find_layer_recursive(func_model, last_conv_layer_name) or find_last_conv_recursive(func_model)
            else:
                func_model = model

            grad_model = keras.models.Model(func_model.inputs, [target_layer.output, func_model.output])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_idx = ops.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            grads = tape.gradient(class_channel, conv_outputs)
        
    elif backend == "torch":
        import torch
        # For Torch, we use Forward Hooks to bypass Functional API connectivity issues
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
        
        # Attach hook to the target layer
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.enable_grad():
                # Standardize input to numpy first to handle cross-framework tensors (TF -> Torch)
                img_numpy = ops.convert_to_numpy(img_array)
                img_tensor = torch.from_numpy(img_numpy).float()
                
                img_tensor.requires_grad = True # CRITICAL: Enable tracking for this image
                
                predictions = model(img_tensor)
                class_idx = ops.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
                
                # activations[0] contains the output of target_layer
                conv_outputs = activations[0]
                grads = torch.autograd.grad(class_channel, conv_outputs)[0]
        finally:
            handle.remove() # Cleanup hook
        
    else:
        raise NotImplementedError(f"Grad-CAM gradient calculation not implemented for backend: {backend}")
    
    # 4. Generate Heatmap using Keras Ops (Backend-agnostic)
    pooled_grads = ops.mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., None]
    heatmap = ops.squeeze(heatmap)
    
    # ReLU and Normalization
    heatmap = ops.maximum(heatmap, 0.0)
    max_val = ops.max(heatmap)
    if max_val != 0:
        heatmap /= max_val
    
    return ops.convert_to_numpy(heatmap)
