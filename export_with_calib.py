#!/usr/bin/env python3
"""
Script to export quantized LaMa model from AMP process to QAI Hub for compilation and profiling.
This script adapts the export.py workflow to work with the quantized ONNX model output from aimet-amp.py.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, cast, Any
import logging

import qai_hub as hub
import onnx
import torch
from PIL import Image
import numpy as np

from qai_hub_models.models.common import ExportResult, Precision, TargetRuntime
from qai_hub_models.models.lama_dilated import Model
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_on_target_demo_cmd,
    print_profile_metrics_from_job,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def export_quantized_model(
    quantized_model_path: str,
    device: Optional[str] = None,
    chipset: Optional[str] = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    target_runtime: TargetRuntime = TargetRuntime.QNN,
    compile_options: str = "",
    profile_options: str = "",
) -> ExportResult:
    """
    Export quantized ONNX model to QAI Hub for compilation and profiling.
    
    Parameters:
        quantized_model_path: Path to the quantized ONNX model from AMP process
        device: Device for which to export the model
        chipset: If set, will choose a random device with this chipset
        skip_profiling: If set, skips profiling of compiled model on real devices
        skip_inferencing: If set, skips computing on-device outputs from sample data
        skip_downloading: If set, skips downloading of compiled model
        skip_summary: If set, skips waiting for and summarizing results
        output_dir: Directory to store generated assets
        target_runtime: Which on-device runtime to target
        compile_options: Additional options to pass when submitting the compile job
        profile_options: Additional options to pass when submitting the profile job
        
    Returns:
        ExportResult containing job metadata
    """
    model_name = "lama_mixed_precision"
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    
    # Setup device
    if not device and not chipset:
        hub_device = hub.Device("Samsung Galaxy S24 (Family)")
        logger.info("Using default device: Samsung Galaxy S24")
    else:
        hub_device = hub.Device(
            name=device or "", attributes=f"chipset:{chipset}" if chipset else []
        )
        logger.info(f"Using device: {hub_device.name}")
    
    # Load the quantized ONNX model
    logger.info(f"Loading quantized ONNX model from {quantized_model_path}")
    if not os.path.exists(quantized_model_path):
        raise FileNotFoundError(f"Quantized model not found at {quantized_model_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(quantized_model_path)
    
    # Get input specification from the original model class
    logger.info("Getting input specifications...")
    original_model = Model.from_pretrained()
    input_spec = original_model.get_input_spec()
    
    # Compile options for quantized model
    model_compile_options = f"--target_runtime {target_runtime.name.lower()}"
    if compile_options:
        model_compile_options += f" {compile_options}"
    
    logger.info(f"Compiling quantized model {model_name} for {target_runtime.name}")
    logger.info(f"Compile options: {model_compile_options}")
    
    # Submit compile job
    submitted_compile_job = hub.submit_compile_job(
        model=onnx_model,
        input_specs=input_spec,
        device=hub_device,
        name=model_name,
        options=model_compile_options,
    )
    compile_job = cast(hub.client.CompileJob, submitted_compile_job)
    
    # Profile the model performance on a real device
    profile_job: Optional[hub.client.ProfileJob] = None
    if not skip_profiling:
        profile_options_all = profile_options
        logger.info(f"Profiling quantized model {model_name} on device {hub_device.name}")
        submitted_profile_job = hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=hub_device,
            name=model_name,
            options=profile_options_all,
        )
        profile_job = cast(hub.client.ProfileJob, submitted_profile_job)
    
    # Run inference on sample inputs
    inference_job: Optional[hub.client.InferenceJob] = None
    if not skip_inferencing:
        logger.info(f"Running inference for quantized {model_name} on device")
        
        # Create sample inputs matching the LaMa model requirements
        sample_inputs = original_model.sample_inputs(input_spec)
        
        submitted_inference_job = hub.submit_inference_job(
            model=compile_job.get_target_model(),
            inputs=sample_inputs,
            device=hub_device,
            name=model_name,
            options=profile_options,
        )
        inference_job = cast(hub.client.InferenceJob, submitted_inference_job)
    
    # Download the compiled model
    if not skip_downloading:
        logger.info(f"Downloading compiled model to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        target_model = compile_job.get_target_model()
        assert target_model is not None
        target_model.download(str(output_path / model_name))
    
    # Summarize results from profiling and inference
    if not skip_summary and profile_job is not None:
        logger.info("Waiting for profiling job to complete...")
        assert profile_job.wait().success, "Profile job failed: " + profile_job.url
        profile_data: dict[str, Any] = profile_job.download_profile()
        print_profile_metrics_from_job(profile_job, profile_data)
    
    if not skip_summary and inference_job is not None:
        logger.info("Waiting for inference job to complete...")
        sample_inputs = original_model.sample_inputs(input_spec, use_channel_last_format=False)
        
        # Get reference outputs from original model
        original_model.eval()
        with torch.no_grad():
            torch_out = original_model(*sample_inputs)
        
        assert inference_job.wait().success, "Inference job failed: " + inference_job.url
        inference_result = inference_job.download_output_data()
        assert inference_result is not None
        
        print_inference_metrics(
            inference_job, inference_result, torch_out, original_model.get_output_names()
        )
    
    if not skip_summary:
        print_on_target_demo_cmd(compile_job, Path(__file__).parent, hub_device)
    
    logger.info("Export to QAI Hub completed successfully!")
    
    return ExportResult(
        compile_job=compile_job,
        inference_job=inference_job,
        profile_job=profile_job,
        quantize_job=None,  # Quantization was done by AMP
    )


def main():
    """Main function to export quantized LaMa model to QAI Hub."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export quantized LaMa model from AMP to QAI Hub"
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        default="./output/lama_mixed_precision.onnx",
        help="Path to quantized ONNX model from AMP process"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Target device name for compilation and profiling"
    )
    parser.add_argument(
        "--chipset",
        type=str,
        help="Target chipset (overrides device argument)"
    )
    parser.add_argument(
        "--target-runtime",
        type=str,
        choices=["tflite", "qnn", "qnn_context_binary", "onnx", "precompiled_qnn_onnx"],
        default="qnn",
        help="Target runtime for compilation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qai_hub_output",
        help="Output directory for compiled model"
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip profiling on device"
    )
    parser.add_argument(
        "--skip-inferencing",
        action="store_true",
        help="Skip inference testing"
    )
    parser.add_argument(
        "--skip-downloading",
        action="store_true",
        help="Skip downloading compiled model"
    )
    parser.add_argument(
        "--compile-options",
        type=str,
        default="",
        help="Additional compile options"
    )
    parser.add_argument(
        "--profile-options",
        type=str,
        default="",
        help="Additional profile options"
    )
    
    args = parser.parse_args()
    
    # Convert target runtime string to enum
    target_runtime_map = {
        "tflite": TargetRuntime.TFLITE,
        "qnn": TargetRuntime.QNN,
        "qnn_context_binary": TargetRuntime.QNN_CONTEXT_BINARY,
        "onnx": TargetRuntime.ONNX,
        "precompiled_qnn_onnx": TargetRuntime.PRECOMPILED_QNN_ONNX,
    }
    target_runtime = target_runtime_map[args.target_runtime]
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    try:
        export_result = export_quantized_model(
            quantized_model_path=args.quantized_model,
            device=args.device,
            chipset=args.chipset,
            skip_profiling=args.skip_profiling,
            skip_inferencing=args.skip_inferencing,
            skip_downloading=args.skip_downloading,
            target_runtime=target_runtime,
            output_dir=args.output_dir,
            compile_options=args.compile_options,
            profile_options=args.profile_options,
        )
        
        logger.info("✅ Successfully exported quantized model to QAI Hub!")
        
        if export_result.compile_job:
            logger.info(f"Compile job URL: {export_result.compile_job.url}")
        if export_result.profile_job:
            logger.info(f"Profile job URL: {export_result.profile_job.url}")
        if export_result.inference_job:
            logger.info(f"Inference job URL: {export_result.inference_job.url}")
            
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
