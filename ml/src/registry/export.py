import torch
import os
import json

def export_model_pt(model, path, metadata=None):
    torch.save(model.state_dict(), path)
    if metadata:
        meta_path = os.path.splitext(path)[0] + '.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

def export_model_onnx(model, sample_input, path, metadata=None):
    torch.onnx.export(model, sample_input, path)
    if metadata:
        meta_path = os.path.splitext(path)[0] + '.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f) 