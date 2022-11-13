"""Image/Video Light Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import todos
from . import llflow

import pdb


def get_tvm_model():
    """
    TVM model base on torch.jit.trace
    """

    model_path = "models/image_light.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = llflow.LLFlow()
    model_load(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


def model_load(model, path):
    """Load model."""

    def reverse_layer_name(n):
        if n.find("flowUpsamplerNet.layers.") < 0:
            return n
        a = n.split(".")
        a[2] = str(20 - int(a[2]))
        n = ".".join(a)

        # skip flowUpsamplerNet.layers.3.affine.fAffine.0.actnorm.bias
        if n.find("actnorm") >= 0:
            return n

        # flowUpsamplerNet.layers.3.affine.fAffine.2.weight -->
        #
        # 'flowUpsamplerNet.layers.16.affine.fAffine.4.stdconv.bias',
        # 'flowUpsamplerNet.layers.16.affine.fFeatures.0.stdconv.weight'
        if (n.find("fAffine") > 0 or n.find("fFeatures") > 0) and (n.find("weight") > 0 or n.find("bias") > 0):
            n = n.replace("weight", "stdconv.weight")
            n = n.replace("bias", "stdconv.bias")
        return n

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    target_state_dict = model.state_dict()

    for n, p in state_dict.items():
        # skip flowUpsamplerNet.f.0.weight etc ...
        if n.find("flowUpsamplerNet.f") >= 0:
            continue

        m = reverse_layer_name(n)
        if m in target_state_dict.keys():
            target_state_dict[m].copy_(p)
        else:
            # print(m)
            raise KeyError(m)

    torch.save(model.state_dict(), "/tmp/image_light.pth")


def get_light_model():
    """Create model."""

    model_path = "models/image_light.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = llflow.LLFlow()
    model_load(model, checkpoint)
    # todos.model.load(model, "/tmp/image_light.pth")
    model = todos.model.ResizePadModel(model)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_light.torch"):
        model.save("output/image_light.torch")

    return model, device


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_light_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
