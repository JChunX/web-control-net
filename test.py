import os
# os.environ['DEBUG'] = '2'
from examples.stable_diffusion import *
from tinygrad.nn.state import load_state_dict, torch_load
from tinygrad.ops import Device

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from cldm import ControlNetModel, ControlledStableDiffusion, ControlledUNetModel

from diffusers.utils import load_image

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

Device.DEFAULT = "WEBGPU"

FILENAME_DIFFUSION = Path(os.path.abspath('')) / "weights/sd-v1-4.ckpt"
download_file('https://huggingface.co/CompVis/stable-diffusion-v-1-5-original/resolve/main/sd-v1-5.ckpt', FILENAME_DIFFUSION)
state_dict_diffusion = torch_load(FILENAME_DIFFUSION)["state_dict"]
diffusion_model = StableDiffusion()
diffusion_model.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model=ControlledUNetModel())
load_state_dict(diffusion_model, state_dict_diffusion, strict=True)

FILENAME_CONTROLNET = Path(os.path.abspath('')) / "weights/sd-controlnet-canny.bin"
download_file('https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin', FILENAME_CONTROLNET)
state_dict_controlnet = torch_load(FILENAME_CONTROLNET)

controlnet = ControlNetModel(cross_attention_dim=768)

load_state_dict(controlnet, state_dict_controlnet, strict=False)

model = ControlledStableDiffusion(diffusion_model, controlnet)


# CLIP etc.
prompt = "a painting of a cat"
seed = 42
steps = 5

tokenizer = ClipTokenizer()
prompt = Tensor([tokenizer.encode(prompt)])
context = model.cond_stage_model.transformer.text_model(prompt).realize()
print("got CLIP context", context.shape)

prompt = Tensor([tokenizer.encode("")])
unconditional_context = model.cond_stage_model.transformer.text_model(prompt).realize()
print("got unconditional CLIP context", unconditional_context.shape)

# done with clip model
del model.cond_stage_model

timesteps = list(range(1, 1000, 1000//steps))
print(f"running for {timesteps} timesteps")
alphas = model.alphas_cumprod[Tensor(timesteps)]
alphas_prev = Tensor([1.0]).cat(alphas[:-1])

# start with random noise
if seed is not None: Tensor._seed = seed
latent = Tensor.randn(1,4,64,64)

canny_condition = np.array(canny_image).transpose(2, 0, 1).astype(np.float32) / 255.0
canny_condition = Tensor(canny_condition).unsqueeze(0)

guidance = 7.5


@TinyJit
def run(model, *x): return model(*x).realize()

timing = True

with Context(BEAM=getenv("LATEBEAM")):
    for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
        GlobalCounters.reset()
        t.set_description("%3d %3d" % (index, timestep))
        with Timing("step in ", enabled=timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
            tid = Tensor([index])
            latent = run(model, canny_condition, unconditional_context, context, latent, Tensor([timestep]), alphas[tid], alphas_prev[tid], Tensor([guidance]), 1.0)
            while (not latent.numpy().any() or (np.isnan(latent.numpy()).any())):
                print("latent: ", latent.numpy().max())
            if timing: Device[Device.DEFAULT].synchronize()
    del run
    
x = model.decode(latent).numpy()
x = np.uint8(np.round(x))
x_image = Image.fromarray(x)
x_image.save("debug.png")
plt.figure(figsize=(10, 10))
plt.imshow(x_image)
plt.show()