from cog import BasePredictor, Input, Path
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os
import subprocess
import uuid

# 定义您的LoRA风格字典
STYLE_TYPE_LORA_DICT = {
    "3D_Chibi": "3D_Chibi_lora_weights.safetensors",
    "American_Cartoon": "American_Cartoon_lora_weights.safetensors",
    "Chinese_Ink": "Chinese_Ink_lora_weights.safetensors",
    "Clay_Toy": "Clay_Toy_lora_weights.safetensors",
    "Fabric": "Fabric_lora_weights.safetensors",
    "Ghibli": "Ghibli_lora_weights.safetensors",
    "Irasutoya": "Irasutoya_lora_weights.safetensors",
    "Jojo": "Jojo_lora_weights.safetensors",
    "Oil_Painting": "Oil_Painting_lora_weights.safetensors",
    "Pixel": "Pixel_lora_weights.safetensors",
    "Snoopy": "Snoopy_lora_weights.safetensors",
    "Poly": "Poly_lora_weights.safetensors",
    "LEGO": "LEGO_lora_weights.safetensors",
    "Origami" : "Origami_lora_weights.safetensors",
    "Pop_Art" : "Pop_Art_lora_weights.safetensors",
    "Van_Gogh" : "Van_Gogh_lora_weights.safetensors",
    "Paper_Cutting" : "Paper_Cutting_lora_weights.safetensors",
    "Line" : "Line_lora_weights.safetensors",
    "Vector" : "Vector_lora_weights.safetensors",
    "Picasso" : "Picasso_lora_weights.safetensors",
    "Macaron" : "Macaron_lora_weights.safetensors",
    "Rick_Morty" : "Rick_Morty_lora_weights.safetensors"
}

class Predictor(BasePredictor):
    def setup(self):
        # 在服务器第一次启动时加载所有模型和设置
        print("--- Starting setup ---")

        # 登录 Hugging Face
        hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hf_token:
            subprocess.run(["huggingface-cli", "login", "--token", hf_token])
            print("Hugging Face login successful.")
        else:
            print("WARNING: Hugging Face token not found.")

        # 加载主模型
        self.pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        )

        # 启用内存优化
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        print("--- Pipeline loaded and optimized ---")

    def predict(
        self,
        image: Path = Input(description="需要转换风格的输入图片"),
        style_name: str = Input(
            description="选择一个艺术风格",
            choices=list(STYLE_TYPE_LORA_DICT.keys()),
            default="3D_Chibi"
        )
    ) -> Path:
        # 运行一次风格转换
        print(f"--- Starting prediction for style: {style_name} ---")

        lora_filename = STYLE_TYPE_LORA_DICT.get(style_name)
        if not lora_filename:
            raise ValueError(f"Invalid style_name: {style_name}")

        # 加载LoRA权重
        # 注意: LoRA权重会从指定的Hugging Face仓库下载
        self.pipeline.load_lora_weights(
            "Owen777/Kontext-Style-Loras",
            weight_name=lora_filename,
            adapter_name="style_lora"
        )
        self.pipeline.set_adapters(["style_lora"], adapter_weights=[1.0])
        print(f"LoRA '{lora_filename}' loaded.")

        origin_image = load_image(str(image))
        original_width, original_height = origin_image.size

        input_image = origin_image.resize((1024, 1024))
        prompt = f"Turn this image into the {style_name.replace('_', ' ')} style."

        print("Generating image...")
        generated_image = self.pipeline(
            image=input_image,
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=24
        ).images[0]
        print("Image generation complete.")

        # 卸载LoRA，为下一次不同的请求做准备
        self.pipeline.disable_lora()
        print("LoRA unloaded.")

        # 生成的图片并保存
        unique_id = uuid.uuid4()
        output_filename = f"{unique_id}.png"
        output_path = f"/tmp/{output_filename}"
        output_image = generated_image.resize((original_width, original_height))
        output_image.save(output_path)

        return Path(output_path)
