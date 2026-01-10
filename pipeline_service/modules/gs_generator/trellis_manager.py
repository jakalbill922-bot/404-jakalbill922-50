from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional
import io

import torch
from PIL import Image, ImageStat

from config import Settings
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline
from schemas import TrellisResult, TrellisRequest, TrellisParams

import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np

class TrellisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.trellis_model_id
        )
        self.pipeline.cuda()
        logger.success("Trellis pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        trellis_request: TrellisRequest,
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"Generating Trellis {trellis_request.seed=} and image size {trellis_request.images[0].size}")

        params = self.default_params.overrided(trellis_request.params)
        buffer = None
        start = time.time()
        try:
            if trellis_request.default:
                outputs = self.pipeline.run(
                    images_rgb[0],
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": 25,  # 25-50 range, higher = better structure but slower
                        "cfg_strength": 5.5,  # 7.0-8.0 range, controls adherence to image
                    },
                    slat_sampler_params={
                        "steps": 50,  # 50-100 range, MOST IMPACT on texture quality
                        "cfg_strength": 2.0,  # 3.0-4.0 range, controls texture detail
                    },
                    preprocess_image=True,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                )
            else:
                # Generate with voxel-aware texture steps
                outputs, num_voxels = self.pipeline.run_multi_image_with_voxel_count(
                    images_rgb,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                    voxel_threshold=25000,
                )
            
            generation_time = time.time() - start
            gaussian = outputs["gaussian"][0]

            # temp_ply = "temp_before_refine.ply"
            # gaussian.save_ply(temp_ply)
            
            # pcd = o3d.io.read_point_cloud(temp_ply)
            # num_points_before = len(pcd.points)
            
           
            # pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
            #     nb_neighbors=10,
            #     std_ratio=2.0
            # )
            
            # num_points_after = len(pcd_filtered.points)
            # removed_points = num_points_before - num_points_after
            
            # logger.warning(f"Outlier removal: {removed_points} points removed ({removed_points/num_points_before*100:.1f}%)")
            
            # # Create boolean mask from indices
            # inlier_mask = np.zeros(num_points_before, dtype=bool)
            # inlier_mask[inlier_indices] = True
            
            # plydata = PlyData.read(temp_ply)
            # vertex = plydata['vertex']
            
            # filtered_vertex = vertex[inlier_mask]
            
            # refined_ply = temp_ply.replace('.ply', '_refined.ply')
            # new_vertex = PlyElement.describe(filtered_vertex, 'vertex')
            # PlyData([new_vertex], text=False).write(refined_ply)
            
            # try:
            #     mask_torch = torch.tensor(inlier_mask, dtype=torch.bool, device=gaussian.device)
                
            #     # # Apply statistical outlier mask
            #     if hasattr(gaussian, '_xyz') and gaussian._xyz is not None:
            #         gaussian._xyz = gaussian._xyz[mask_torch]
            #     if hasattr(gaussian, '_features_dc') and gaussian._features_dc is not None:
            #         gaussian._features_dc = gaussian._features_dc[mask_torch]
            #     if hasattr(gaussian, '_features_rest') and gaussian._features_rest is not None:
            #         gaussian._features_rest = gaussian._features_rest[mask_torch]
            #     if hasattr(gaussian, '_scaling') and gaussian._scaling is not None:
            #         gaussian._scaling = gaussian._scaling[mask_torch]
            #     if hasattr(gaussian, '_rotation') and gaussian._rotation is not None:
            #         gaussian._rotation = gaussian._rotation[mask_torch]
            #     if hasattr(gaussian, '_opacity') and gaussian._opacity is not None:
            #         gaussian._opacity = gaussian._opacity[mask_torch]
                
            #     # Effective fog removal with object protection
            #     if hasattr(gaussian, '_opacity') and gaussian._opacity is not None:
            #         opacity_values = torch.sigmoid(gaussian._opacity).squeeze()
                    
            #         # Initialize mask - keep everything by default
            #         keep_mask = torch.ones_like(opacity_values, dtype=torch.bool)
                    
            #         if hasattr(gaussian, '_xyz') and gaussian._xyz is not None and hasattr(gaussian, '_scaling') and gaussian._scaling is not None:
            #             xyz = gaussian._xyz
                    
            #             x_coords = xyz[:, 0]  # X coordinate (horizontal axis)
            #             x_min = x_coords.min()
            #             x_max = x_coords.max()
            #             print(f"X range: min={x_min.item():.3f}, max={x_max.item():.3f}")
            #             x_range = x_max - x_min
            #             is_x_plane = (x_coords < (x_min + x_range * 0.001)) | (x_coords > (x_max - x_range * 0.001))   
                        
            #             y_coords = xyz[:, 1]  # Y coordinate (horizontal axis)
            #             y_min = y_coords.min()
            #             y_max = y_coords.max()
            #             print(f"Y range: min={y_min.item():.3f}, max={y_max.item():.3f}")
            #             y_range = y_max - y_min
            #             is_y_plane = (y_coords < (y_min + y_range * 0.001)) | (y_coords > (y_max - y_range * 0.001))   
                        
            #             z_coords = xyz[:, 2]  # Z coordinate (vertical axis)
            #             z_min = z_coords.min()
            #             z_max = z_coords.max()
            #             print(f"Z range: min={z_min.item():.3f}, max={z_max.item():.3f}")
            #             z_range = z_max - z_min
            #             # Remove points in bottom 5% of vertical range
            #             is_z_plane = (z_coords < (z_min + z_range * 0.005)) | (z_coords > (z_max - z_range * 0.002))
                        
            #             remove_mask = is_x_plane | is_y_plane | is_z_plane
                        
            #             keep_mask = ~remove_mask
                    
            #         num_removed_fog = (~keep_mask).sum().item()
            #         if num_removed_fog > 0:
            #             logger.warning(f"Fog + ground plane removal (7 patterns): {num_removed_fog} Gaussians removed ({num_removed_fog/len(keep_mask)*100:.1f}%)")
                        
            #             gaussian._xyz = gaussian._xyz[keep_mask]
            #             gaussian._features_dc = gaussian._features_dc[keep_mask]
            #             if gaussian._features_rest is not None:
            #                 gaussian._features_rest = gaussian._features_rest[keep_mask]
            #             gaussian._scaling = gaussian._scaling[keep_mask]
            #             gaussian._rotation = gaussian._rotation[keep_mask]
            #             gaussian._opacity = gaussian._opacity[keep_mask]
                        
            # except Exception as e:
            #     logger.warning(f"Error: {e}")

            # # Clean up temporary files
            # if os.path.exists(temp_ply):
            #     os.remove(temp_ply)
            # if os.path.exists(refined_ply):
            #     os.remove(refined_ply)


            # Save ply to buffer
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None # bytes
            )

            # logger.success(f"Trellis finished generation in {generation_time:.2f}s with {num_voxels} occupied voxels.")
            return result
        finally:
            if buffer:
                buffer.close()

