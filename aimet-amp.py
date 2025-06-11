#!/usr/bin/env python3
"""
Script to apply Automatic Mixed Precision (AMP) quantization to LaMa model using AIMET-ONNX.
This script is adapted from the AMP.ipynb notebook to work with the LaMa image inpainting model.
"""

import os
import glob
import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Tuple, List
import logging

# AIMET imports
from aimet_common.defs import QuantScheme, CallbackFunc, QuantizationDataType
from aimet_common.amp.utils import AMPSearchAlgo
import aimet_onnx
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.mixed_precision import choose_mixed_precision

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Explicitly add console handler
    ]
)
logger = logging.getLogger(__name__)

class LaMaDataset(Dataset):
    """Dataset class for LaMa test images with masks."""
    
    def __init__(self, data_dir: str, image_size: int = 512):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Find all image files (non-mask files)
        all_files = glob.glob(os.path.join(data_dir, "*.png"))
        self.image_files = [f for f in all_files if not f.endswith("_mask.png")]
        
        # Verify corresponding mask files exist
        self.valid_pairs = []
        for img_file in self.image_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            mask_file = os.path.join(data_dir, f"{base_name}_mask.png")
            if os.path.exists(mask_file):
                self.valid_pairs.append((img_file, mask_file))
        
        logger.info(f"Found {len(self.valid_pairs)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Load and preprocess mask
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.image_size, self.image_size))
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        
        # Convert to torch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask, os.path.basename(img_path)

class LaMaDataPipeline:
    """Data pipeline for LaMa model evaluation."""
    
    def __init__(self, data_dir: str, batch_size: int = 1, image_size: int = 512):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        
    def get_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for LaMa dataset."""
        dataset = LaMaDataset(self.data_dir, self.image_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
    
    @staticmethod
    def evaluate_inpainting_quality(session: ort.InferenceSession, data_loader: DataLoader, max_samples: int = None) -> float:
        """
        Evaluate inpainting quality using PSNR metric.
        This is a simplified evaluation - in practice, you might want to use more sophisticated metrics.
        """
        total_psnr = 0.0
        num_samples = 0
        
        input_names = [inp.name for inp in session.get_inputs()]
        
        for batch_idx, (images, masks, filenames) in enumerate(data_loader):
            if max_samples and num_samples >= max_samples:
                break
                
            batch_size = images.shape[0]
            
            # Prepare inputs for ONNX model
            inputs = {
                input_names[0]: images.numpy(),  # image
                input_names[1]: masks.numpy()    # mask
            }
            
            # Run inference
            try:
                outputs = session.run(None, inputs)
                painted_images = outputs[0]
                
                # Calculate PSNR for each image in batch
                for i in range(batch_size):
                    original = images[i].numpy()
                    painted = painted_images[i]
                    
                    # Calculate PSNR (simplified - using the painted regions)
                    mask_np = masks[i, 0].numpy()
                    masked_regions = mask_np > 0.5
                    
                    if np.any(masked_regions):
                        mse = np.mean((original[:, masked_regions] - painted[:, masked_regions]) ** 2)
                        if mse > 0:
                            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                        else:
                            psnr = 100  # Perfect reconstruction
                        total_psnr += psnr
                        num_samples += 1
                    
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if num_samples == 0:
            return 0.0
        
        avg_psnr = total_psnr / num_samples
        logger.info(f"Average PSNR: {avg_psnr:.2f} dB on {num_samples} samples")
        return avg_psnr

def pass_calibration_data(session: ort.InferenceSession, data_loader: DataLoader, max_samples: int = 1000):
    """Pass calibration data through the model for computing encodings."""
    input_names = [inp.name for inp in session.get_inputs()]
    
    sample_count = 0
    for images, masks, _ in data_loader:
        if sample_count >= max_samples:
            break
            
        inputs = {
            input_names[0]: images.numpy(),
            input_names[1]: masks.numpy()
        }
        
        session.run(None, inputs)
        sample_count += images.shape[0]
        
        if sample_count % 100 == 0:
            logger.info(f"Processed {sample_count} calibration samples")



def main():
    # Configuration
    DATASET_DIR = './LaMa_test_images'
    MODEL_PATH = './lama_dilated.onnx'
    RESULTS_DIR = './amp_results'
    BATCH_SIZE = 1
    IMAGE_SIZE = 512
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load the ONNX model
    logger.info("Loading ONNX model...")
    model = onnx.load(MODEL_PATH)
    
    # Simplify the model (optional but recommended)
    try:
        from onnxsim import simplify
        model, _ = simplify(model)
        logger.info("Model simplified successfully")
    except Exception as e:
        logger.warning(f"ONNX Simplifier failed: {e}. Proceeding with original model")
    
    # Setup providers for ONNX Runtime
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        logger.info("Using CUDA execution provider")
    else:
        providers = ['CPUExecutionProvider']
        logger.info("Using CPU execution provider")
    
    # Create data pipeline
    data_pipeline = LaMaDataPipeline(DATASET_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    data_loader = data_pipeline.get_dataloader()
    
    # Evaluate baseline FP32 accuracy
    logger.info("Evaluating baseline FP32 accuracy...")
    fp32_session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    baseline_psnr = LaMaDataPipeline.evaluate_inpainting_quality(fp32_session, data_loader)
    logger.info(f"Baseline FP32 PSNR: {baseline_psnr:.2f} dB")
    
    # Fold Batch Normalization layers
    logger.info("Folding Batch Normalization layers...")
    try:
        _ = fold_all_batch_norms_to_weight(model)
        logger.info("Batch normalization folding completed")
    except Exception as e:
        logger.warning(f"Batch norm folding failed: {e}")
    
    # Create Quantization Simulation Model
    logger.info("Creating Quantization Simulation Model...")
    sim = QuantizationSimModel(model=model,
                               quant_scheme=QuantScheme.min_max,
                               default_param_bw=8,
                               default_activation_bw=8,
                               default_data_type=QuantizationDataType.int,
                               providers=providers)
    
    # Compute encodings (calibration)
    logger.info("Computing quantization encodings...")
    def calibration_callback(session, args=None):
        pass_calibration_data(session, data_loader, max_samples=500)
    
    sim.compute_encodings(forward_pass_callback=calibration_callback)
    
    # Evaluate quantized model accuracy
    logger.info("Evaluating quantized model accuracy...")
    quantized_psnr = LaMaDataPipeline.evaluate_inpainting_quality(sim.session, data_loader)
    logger.info(f"Quantized INT8 PSNR: {quantized_psnr:.2f} dB")
    
    # Define AMP candidates
    candidates = [
        ((16, QuantizationDataType.int), (16, QuantizationDataType.int)), 
        ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
        ((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
        ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
    ]
    
    # Setup callback functions for AMP
    forward_pass_callback = CallbackFunc(calibration_callback, func_callback_args=None)
    
    # Phase 1 evaluation: Use PSNR with reduced sample size for faster evaluation
    eval_callback_for_phase1 = CallbackFunc(
        lambda session, args: LaMaDataPipeline.evaluate_inpainting_quality(session, data_loader, max_samples=200),
        func_callback_args=None
    )
    
    # Phase 2 evaluation: Use full accuracy evaluation
    eval_callback_for_phase2 = CallbackFunc(
        lambda session, args: LaMaDataPipeline.evaluate_inpainting_quality(session, data_loader),
        func_callback_args=None
    )
    
    # AMP algorithm parameters
    allowed_accuracy_drop = 1.0  # Allow 1 dB drop in PSNR
    amp_search_algo = AMPSearchAlgo.Binary
    
    logger.info("Starting AMP algorithm...")
    logger.info(f"Baseline PSNR: {baseline_psnr:.2f} dB")
    logger.info(f"Allowed accuracy drop: {allowed_accuracy_drop} dB")
    
    # Run AMP algorithm
    try:
        pareto_front_list = choose_mixed_precision(
            sim, 
            candidates,
            eval_callback_for_phase1=eval_callback_for_phase1, 
            eval_callback_for_phase2=eval_callback_for_phase2, 
            allowed_accuracy_drop=allowed_accuracy_drop, 
            results_dir=RESULTS_DIR,
            clean_start=True, 
            forward_pass_callback=forward_pass_callback,
            amp_search_algo=amp_search_algo,
            phase1_optimize=True
        )
        
        logger.info("AMP algorithm completed successfully!")
        
        if pareto_front_list is not None:
            logger.info(f"Pareto front contains {len(pareto_front_list)} points")
        else:
            logger.info("Algorithm early-exited: all candidates within acceptable accuracy range")
        
        # Evaluate final mixed precision model
        final_psnr = LaMaDataPipeline.evaluate_inpainting_quality(sim.session, data_loader)
        logger.info(f"Final mixed precision PSNR: {final_psnr:.2f} dB")
        
        # Export the optimized model
        output_dir = './output'
        os.makedirs(output_dir, exist_ok=True)
        sim.export(path=output_dir, filename_prefix='lama_mixed_precision')
        logger.info(f"Mixed precision model exported to {output_dir}")
        
    except Exception as e:
        logger.error(f"AMP algorithm failed: {e}")
        raise

if __name__ == "__main__":
    main() 