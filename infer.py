import os
import sys
import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
from wan.models.face_align import FaceAlignment
from wan.models.face_model import FaceModel
from wan.models.pdf import FanEncoder, det_landmarks, get_drive_expression_pd_fgc
from wan.models.portrait_encoder import PortraitEncoder
from wan.pipeline.pipeline_wan_long import WanI2VLongPipeline

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from wan.dist import set_multi_gpus_devices, shard_model
from wan.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel)
from wan.models.cache_utils import get_teacache_coefficients
from wan.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from wan.utils.lora_utils import merge_lora, unmerge_lora
from wan.utils.utils import (filter_kwargs, get_image_to_video_latent,
                             save_videos_grid, simple_save_videos_grid)
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from xfuser.core.distributed import (get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group, get_world_group, init_distributed_environment, initialize_model_parallel)
import torch.distributed as dist

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
#
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory.
#
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
#
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
# and the transformer model has been quantized to float8, which can save more GPU memory.
#
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_full_load"  # [model_full_load, model_cpu_offload, sequential_cpu_offload, model_full_load_and_qfloat8, model_cpu_offload_and_qfloat8]
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used.
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = False
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process,
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold | Model Name          | threshold |
# | Wan2.1-T2V-1.3B     | 0.05~0.10 | Wan2.1-T2V-14B      | 0.10~0.15 | Wan2.1-I2V-14B-720P | 0.20~0.30 |
# | Wan2.1-I2V-14B-480P | 0.20~0.25 | Wan2.1-Fun-*-1.3B-* | 0.05~0.10 | Wan2.1-Fun-*-14B-*  | 0.20~0.30 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
wan_model_name          = "/path/FlashPortrait/checkpoints/Wan2.1-I2V-14B-720P"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics.
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 5

# Load pretrained model if need
transformer_path    = "/path/FlashPortrait/checkpoints/FlashPortrait/transformer.pt"
portrait_encoder_path = "/path/FlashPortrait/checkpoints/FlashPortrait/portrait_encoder.pt"
det_model_path = "/path/FlashPortrait/checkpoints/FlashPortrait/face_det.onnx"
alignment_model_path = "/path/FlashPortrait/checkpoints/FlashPortrait/face_landmark.onnx"
pd_fpg_model_path = "/path/FlashPortrait/checkpoints/FlashPortrait/pd_fpg.pth"
vae_path            = None
lora_path           = None

# Other params
sample_size         = [512, 512]
max_size            = 720
sub_num_frames      = 201
latents_num_frames  = 51
context_overlap     = 30
context_size        = 51
ip_scale            = 1.0
text_cfg_scale      = 1.0
emo_cfg_scale       = 4.0  # options: [2, 3, 4]
fps                 = 25

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start  = "/path/FlashPortrait/examples/case-1/reference.png"
validation_driven_video_path = "/path/FlashPortrait/examples/case-1/driven_video.mp4"

# prompts
prompt              = "The girl is singing"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
seed                = 42
num_inference_steps = 30
lora_weight         = 0.55
save_path           = "samples/wan-videos-i2v"

if ulysses_degree > 1 or ring_degree > 1:
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if get_sp_group is None:
        raise RuntimeError("xfuser is not installed.")
    dist.init_process_group("nccl")
    print('parallel inference enabled: ulysses_degree=%d ring_degree=%d rank=%d world_size=%d' % (ulysses_degree, ring_degree, dist.get_rank(), dist.get_world_size()))
    assert dist.get_world_size() == ring_degree * ulysses_degree, "number of GPUs(%d) should be equal to ring_degree * ulysses_degree." % dist.get_world_size()
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(), ring_degree=ring_degree, ulysses_degree=ulysses_degree)
    # device = torch.device("cuda:%d" % dist.get_rank())
    device = torch.device(f"cuda:{get_world_group().local_rank}")
    print('rank=%d device=%s' % (get_world_group().rank, str(device)))
else:
    device = "cuda"

# device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)



transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(wan_model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

print(f"From checkpoint: {transformer_path}")
if transformer_path.endswith("safetensors"):
    from safetensors.torch import load_file, safe_open
    transformer_state_dict = load_file(transformer_path)
else:
    transformer_state_dict = torch.load(transformer_path, map_location="cpu", weights_only=True)
transformer_state_dict = transformer_state_dict["state_dict"] if "state_dict" in transformer_state_dict else transformer_state_dict
m, u = transformer.load_state_dict(transformer_state_dict, strict=False)
print(f"portrait transformer missing keys: {len(m)}, unexpected keys: {len(u)}")
print(f"portrait transformer missing keys: {m}")
print(f"portrait transformer unexpected keys: {u}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(wan_model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(wan_model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(wan_model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Clip Image Encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(wan_model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

face_aligner = FaceModel(
    face_alignment_module=FaceAlignment(
        gpu_id=None,
        alignment_model_path=alignment_model_path,
        det_model_path=det_model_path,
    ),
    reset=False,
)
pd_fpg_motion = FanEncoder()
pd_fpg_checkpoint = torch.load(pd_fpg_model_path, map_location="cpu")
m, u = pd_fpg_motion.load_state_dict(pd_fpg_checkpoint, strict=False)
pd_fpg_motion = pd_fpg_motion.eval()

portrait_encoder_state_dict = torch.load(portrait_encoder_path, map_location="cpu", weights_only=True)
proj_prefix = "proj_model."
mouth_prefix = "mouth_proj_model."
emo_prefix = "emo_proj_model."
portrait_encoder_state_dict_sub_proj_state = {}
portrait_encoder_state_dict_sub_mouth_state = {}
portrait_encoder_state_dict_sub_emo_state = {}
for k, v in portrait_encoder_state_dict.items():
    if k.startswith(proj_prefix):
        new_key = k[len(proj_prefix):]  # remove prefix + dot
        portrait_encoder_state_dict_sub_proj_state[new_key] = v
    elif k.startswith(mouth_prefix):
        new_key = k[len(mouth_prefix):]  # remove prefix + dot
        portrait_encoder_state_dict_sub_mouth_state[new_key] = v
    elif k.startswith(emo_prefix):
        new_key = k[len(emo_prefix):]  # remove prefix + dot
        portrait_encoder_state_dict_sub_emo_state[new_key] = v
portrait_encoder = PortraitEncoder(adapter_in_dim=768, adapter_proj_dim=2048)
portrait_encoder.proj_model.load_state_dict(portrait_encoder_state_dict_sub_proj_state)
portrait_encoder.mouth_proj_model.load_state_dict(portrait_encoder_state_dict_sub_mouth_state)
portrait_encoder.emo_proj_model.load_state_dict(portrait_encoder_state_dict_sub_emo_state)
portrait_encoder = portrait_encoder.eval()


# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanI2VLongPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
    portrait_encoder=portrait_encoder,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(wan_model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

def find_replacement(a):
    while a > 0:
        if (a - 1) % 4 == 0:
            return a
        a -= 1
    return 0

def get_emo_feature(video_path, face_aligner, pd_fpg_motion, device=torch.device("cuda")):
    pd_fpg_motion = pd_fpg_motion.to(device)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        resized_frame = frame
        frame_list.append(resized_frame.copy())
        ret, frame = cap.read()
    cap.release()
    num_frames = len(frame_list)
    num_frames = find_replacement(num_frames)
    frame_list = frame_list[:num_frames]
    landmark_list = det_landmarks(face_aligner, frame_list)[1]
    emo_list = get_drive_expression_pd_fgc(pd_fpg_motion, frame_list, landmark_list, device)
    emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]
        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)
        emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)
    emo_feat_all = torch.cat(emo_feat_list, dim=0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0)
    return emo_feat_all, head_emo_feat_all, fps, num_frames

with torch.no_grad():
    image_start = clip_image = Image.open(validation_image_start).convert("RGB")
    width, height = image_start.size
    scale = max_size / max(width, height)
    width, height = (int(width * scale), int(height * scale))
    height_division_factor = 16
    width_division_factor = 16
    if height % height_division_factor != 0:
        height = (height + height_division_factor - 1) // height_division_factor * height_division_factor
        print(f"The height cannot be evenly divided by {height_division_factor}. We round it up to {height}.")
    if width % width_division_factor != 0:
        width = (width + width_division_factor - 1) // width_division_factor * width_division_factor
        print(f"The width cannot be evenly divided by {width_division_factor}. We round it up to {width}.")
    image_start = image_start.resize([width, height], Image.LANCZOS)
    clip_image = clip_image.resize([width, height], Image.LANCZOS)
    input_video = torch.tile(torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), [1, 1, sub_num_frames, 1, 1]) / 255
    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, 1:, ] = 255
    emo_feat_all, head_emo_feat_all, fps, num_frames = get_emo_feature(validation_driven_video_path, face_aligner, pd_fpg_motion, device=device)
    emo_feat_all, head_emo_feat_all = emo_feat_all.unsqueeze(0), head_emo_feat_all.unsqueeze(0)

    sample = pipeline(
        prompt,
        num_frames=num_frames,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=4.0,
        num_inference_steps=num_inference_steps,

        video=input_video,
        mask_video=input_video_mask,
        clip_image=clip_image,
        shift=shift,

        context_overlap=context_overlap,
        context_size=context_size,
        latents_num_frames=latents_num_frames,
        ip_scale=ip_scale,
        head_emo_feat_all=head_emo_feat_all.to(device),
        sub_num_frames=sub_num_frames,
        text_cfg_scale=text_cfg_scale,
        emo_cfg_scale=emo_cfg_scale,
    ).videos
    sample = sample[:, :, 1:]

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if num_frames == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

def simple_save_video():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    simple_save_videos_grid(sample, video_path, fps=fps)


if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        # save_results()
        simple_save_video()
else:
    # save_results()

    simple_save_video()
