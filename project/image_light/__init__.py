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
import time
from tqdm import tqdm
import torch

import redos
import todos
from . import llflow

import pdb


LIGHT_ZEROPAD_TIMES = 8


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

    # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    target_state_dict = model.state_dict()

    for n, p in state_dict.items():
        # skip flowUpsamplerNet.f.0.weight etc ...
        if n.find('flowUpsamplerNet.f') >= 0:
            continue

        m = reverse_layer_name(n)
        if m in target_state_dict.keys():
            target_state_dict[m].copy_(p)
        else:
            # print(m)
            raise KeyError(m)

    torch.save(model.state_dict(), "/tmp/image_light.pth")


def get_model():
    """Create model."""

    model_path = "models/image_light.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = llflow.LLFlow()
    model_load(model, checkpoint)
    # todos.model.load(model, "/tmp/image_light.pth")
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_light.torch"):
        model.save("output/image_light.torch")

    return model, device


def model_forward(model, device, input_tensor):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % LIGHT_ZEROPAD_TIMES != 0 or W % LIGHT_ZEROPAD_TIMES != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=LIGHT_ZEROPAD_TIMES)
    output_tensor = todos.model.forward(model, device, input_tensor)

    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.light(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  clean {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_light", do_service, host, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

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
        predict_tensor = model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  light {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def light_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=light_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.light(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_light", video_service, host, port)
