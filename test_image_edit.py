"""
Standalone test script for Qwen Image Edit utility
Tests the image editing capabilities by generating different views of an object.

Usage:
    python test_image_edit.py <input_image_path> [--output-dir <dir>] [--seed <int>]

Example:
    python test_image_edit.py input.png --output-dir test_outputs --seed 42
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from PIL import Image

# Temporarily disable .env loading to avoid conflicts
os.environ['SETTINGS_DISABLE_DOTENV'] = '1'

# Add pipeline_service to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline_service"))

# Import and configure settings without .env
from config.settings import Settings
from modules.image_edit.qwen_edit_module import QwenEditModule
from logger_config import logger

# Create minimal settings for testing
test_settings = Settings(
    qwen_gpu=0,
    dtype="bf16",
    qwen_edit_base_model_path="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    qwen_edit_model_path="Qwen/Qwen-Image-Edit-2511",
    qwen_edit_height=1024,
    qwen_edit_width=1024,
    num_inference_steps=4,
    true_cfg_scale=1.0,
)


async def test_image_edit(
    input_image_path: str,
    output_dir: str = "test_outputs",
    seed: int = 42
):
    """
    Test Qwen Image Edit module with an input image.
    
    Args:
        input_image_path: Path to the input image
        output_dir: Directory to save output images
        seed: Random seed for reproducibility
    """
    
    # Validate input image
    input_path = Path(input_image_path)
    if not input_path.exists():
        logger.error(f"Input image not found: {input_image_path}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_path}")
    
    # Initialize Qwen module
    logger.info("Initializing Qwen Edit module...")
    qwen = QwenEditModule(test_settings)
    await qwen.startup()
    
    # Load input image
    input_image = Image.open(input_path).convert("RGB")
    logger.info(f"Loaded input image: {input_image.size}")
    
    # Save original input
    input_output_path = output_path / "00_input_original.png"
    input_image.save(input_output_path)
    logger.success(f"Saved: {input_output_path}")
    
    # Define test prompts for different views
    test_cases = [
        {
            "name": "left_three_quarters_view",
            "prompt": "Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "01_edited_left_view.png"
        },
        {
            "name": "right_three_quarters_view",
            "prompt": "Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "02_edited_right_view.png"
        },
        {
            "name": "back_view",
            "prompt": "Show this object in back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "03_edited_back_view.png"
        },
        {
            "name": "front_view",
            "prompt": "Show this object in front view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "04_edited_front_view.png"
        },
        {
            "name": "top_view",
            "prompt": "Show this object in top view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            "filename": "05_edited_top_view.png"
        }
    ]
    
    # Generate edited images for each view
    logger.info(f"\nGenerating {len(test_cases)} different views (seed={seed})...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] Generating: {test_case['name']}")
        logger.info(f"Prompt: {test_case['prompt'][:80]}...")
        
        try:
            # Edit the image
            edited_image = qwen.edit_image(
                prompt_image=input_image,
                seed=seed,
                prompt=test_case["prompt"]
            )
            
            # Save the result
            output_file = output_path / test_case["filename"]
            edited_image.save(output_file)
            logger.success(f"Saved: {output_file}\n")
            
        except Exception as e:
            logger.error(f"Error generating {test_case['name']}: {e}\n")
            continue
    
    # Shutdown
    await qwen.shutdown()
    
    logger.success(f"\n{'='*60}")
    logger.success(f"Test completed!")
    logger.success(f"All outputs saved to: {output_path}")
    logger.success(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen Image Edit module to generate different views of an object"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input image"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_outputs",
        help="Output directory for generated images (default: test_outputs)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_image_edit(
        input_image_path=args.input_image,
        output_dir=args.output_dir,
        seed=args.seed
    ))


if __name__ == "__main__":
    main()
