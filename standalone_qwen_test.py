"""
Standalone Qwen Image Edit Test Script
Completely independent from pipeline - loads model directly and tests editing capabilities.

Usage:
    python standalone_qwen_test.py <input_image_path> [--output-dir <dir>] [--seed <int>]

Example:
    python standalone_qwen_test.py cr7.png --output-dir test_outputs --seed 42
"""

import argparse
import math
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel


def prepare_image(image: Image.Image, megapixels: float = 1.0) -> Image.Image:
    """Resize image to target megapixels while maintaining aspect ratio."""
    total = int(megapixels * 1024 * 1024)
    scale_by = math.sqrt(total / (image.width * image.height))
    width = round(image.width * scale_by)
    height = round(image.height * scale_by)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def load_qwen_pipeline(device="cuda:0", dtype=torch.bfloat16):
    """Load Qwen-2511 model with 4-step Lightning LoRA."""
    print("Loading Qwen-2511 transformer...")
    
    # Model paths
    model_path = "Qwen/Qwen-Image-Edit-2511"
    lora_repo = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    lora_weights = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    
    # Load transformer
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype
    )
    
    # Scheduler config (optimized for 4-step Lightning)
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    print("Loading Qwen-2511 pipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=dtype
    )
    
    print(f"Loading LoRA weights: {lora_weights}...")
    pipeline.load_lora_weights(lora_repo, weight_name=lora_weights)
    
    print(f"Moving to {device}...")
    pipeline = pipeline.to(device)
    
    print("Pipeline ready!")
    return pipeline


def edit_image(pipeline, image: Image.Image, prompt: str, seed: int, device="cuda:0"):
    """Edit image with Qwen pipeline."""
    # Prepare image
    prepared_image = prepare_image(image, megapixels=1.0)
    
    # Generate
    generator = torch.Generator(device=device).manual_seed(seed)
    
    result = pipeline(
        image=prepared_image,
        prompt=prompt,
        generator=generator,
        num_inference_steps=4,
        true_cfg_scale=1.0,
        height=1024,
        width=1024,
    )
    
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Qwen-2511 Image Edit Test"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="test_outputs", 
                       help="Output directory (default: test_outputs)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (default: cuda:0)")
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image not found: {args.input_image}")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Qwen-2511 Image Edit Test")
    print(f"{'='*60}")
    print(f"Input: {args.input_image}")
    print(f"Output: {output_path}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load pipeline
    pipeline = load_qwen_pipeline(device=args.device)
    
    # Load input image
    print(f"\nLoading input image...")
    input_image = Image.open(input_path).convert("RGB")
    print(f"Image size: {input_image.size}")
    
    # Save original
    input_output = output_path / "00_input_original.png"
    input_image.save(input_output)
    print(f"Saved: {input_output}")
    
    # Define test prompts
    test_cases = [
        {
            "name": "Left Three-Quarters View",
            "prompt": "Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "01_edited_left_view.png"
        },
        {
            "name": "Right Three-Quarters View",
            "prompt": "Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "02_edited_right_view.png"
        },
        {
            "name": "Back View",
            "prompt": "Show this object in back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "03_edited_back_view.png"
        }

    ]
    
    # Generate edited views
    print(f"\n{'='*60}")
    print(f"Generating {len(test_cases)} different views...")
    print(f"{'='*60}\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {test_case['name']}")
        
        try:
            edited_image = edit_image(
                pipeline,
                input_image,
                test_case["prompt"],
                args.seed,
                args.device
            )
            
            output_file = output_path / test_case["filename"]
            edited_image.save(output_file)
            print(f"✓ Saved: {output_file}\n")
            
        except Exception as e:
            print(f"✗ Error: {e}\n")
            continue
    
    print(f"{'='*60}")
    print(f"Test completed!")
    print(f"All outputs: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
