from __future__ import annotations
import jinja2
import os
import warnings
from pathlib import Path
from typing import Any, Optional, cast

import qai_hub as hub
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from datasets import load_dataset
import itertools

from qai_hub_models.models.common import ExportResult, Precision, TargetRuntime
from qai_hub_models.models.lama_dilated import Model
from qai_hub_models.utils import quantization as quantization_utils
from qai_hub_models.utils.args import (
    export_parser,
    get_input_spec_kwargs,
    get_model_kwargs,
    validate_precision_runtime,
)
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_on_target_demo_cmd,
    print_profile_metrics_from_job,
)
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
    export_without_hub_access,
)

def indent_except_first(s, width=4):
    lines = s.splitlines()
    if not lines:
        return ''
    first, rest = lines[0], lines[1:]
    indent_str = ' ' * width
    return '\n'.join([first] + [indent_str + line for line in rest])

jinja2.filters.FILTERS['indent_except_first'] = indent_except_first

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# class LocalImageCalibDataset(Dataset):
#     def __init__(self, folder_path: str, transform=None):
#         self.folder_path = folder_path
#         self.transform = transform or T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor(),
#         ])
#         self.image_files = sorted([
#             f for f in os.listdir(folder_path)
#             if f.endswith(".png") and not f.endswith("_mask.png")
#         ])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.folder_path, self.image_files[idx])
#         img = Image.open(img_path).convert("RGB")
#         return self.transform(img).unsqueeze(0)

class LocalImageCalibDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        self.folder_path = folder_path
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])
        self.image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(".png") and not f.endswith("_mask.png")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        mask_filename = image_filename.replace(".png", "_mask.png")

        image_path = os.path.join(self.folder_path, image_filename)
        mask_path = os.path.join(self.folder_path, mask_filename)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.transform(image)
        mask_tensor = self.transform(mask)

        return {"image": image_tensor, "mask": mask_tensor}

        calib_loader = DataLoader(dataset, batch_size=1)

        input_keys = list(input_spec.keys())
        calib_tensors_dict = {key: [] for key in input_keys}

        for i, batch in enumerate(itertools.islice(iter(calib_loader), num_calibration_samples)):
            try:
                if isinstance(batch, dict):
                    for k in input_keys:
                        val = batch[k]
                        assert isinstance(val, torch.Tensor), f"{k} at index {i} is not Tensor"
                        calib_tensors_dict[k].append(val)
                elif isinstance(batch, (list, tuple)) and len(input_keys) > 1:
                    assert len(batch) == len(input_keys), f"Batch {i} size mismatch with input_keys"
                    for k, val in zip(input_keys, batch):
                        assert isinstance(val, torch.Tensor), f"{k} at index {i} is not Tensor"
                        calib_tensors_dict[k].append(val)
                else:
                    k = input_keys[0]
                    val = batch if not isinstance(batch, (list, tuple)) else batch[0]
                    assert isinstance(val, torch.Tensor), f"{k} at index {i} is not Tensor"
                    calib_tensors_dict[k].append(val)
            except Exception as e:
                print(f"[Warning] Skipping sample {i}: {e}")

        lengths = [len(v) for v in calib_tensors_dict.values()]
        assert len(set(lengths)) == 1, f"Inconsistent calibration sample counts: {lengths}"

        calibration_data = calib_tensors_dict

        quantize_job = hub.submit_quantize_job(
            model=onnx_compile_job.get_target_model(),
            calibration_data=calibration_data,
            activations_dtype=precision.activations_type,
            weights_dtype=precision.weights_type,
            name=model_name,
            options=model.get_hub_quantize_options(precision),
        )

class HFCalibDataset(Dataset):
    def __init__(self, hf_name: str, split: str = "train", transform=None):
        self.dataset = load_dataset(hf_name, split=split)
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        return self.transform(img).unsqueeze(0)

def export_model(
    device: Optional[str] = None,
    chipset: Optional[str] = None,
    precision: Precision = Precision.float,
    num_calibration_samples: int | None = None,
    skip_compiling: bool = False,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    compile_options: str = "",
    profile_options: str = "",
    fetch_static_assets: bool = False,
    calib_data_dir: Optional[str] = None,
    calib_hf_dataset: Optional[str] = None,
    **additional_model_kwargs,
) -> ExportResult | list[str]:
    model_name = "lama_dilated"
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    hub_device = hub.Device(
        name=device or "", attributes=f"chipset:{chipset}" if chipset else []
    ) if device or chipset else hub.Device("Samsung Galaxy S24 (Family)")

    if fetch_static_assets or not can_access_qualcomm_ai_hub():
        return export_without_hub_access(
            model_name,
            "LaMa-Dilated",
            hub_device.name or f"Device (Chipset {chipset})",
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            precision,
            compile_options,
            profile_options,
            is_forced_static_asset_fetch=fetch_static_assets,
        )

    use_channel_last_format = target_runtime.channel_last_native_execution

    model = Model.from_pretrained(**get_model_kwargs(Model, additional_model_kwargs))
    input_spec = model.get_input_spec(
        **get_input_spec_kwargs(model, additional_model_kwargs)
    )

    quantize_job = None
    if precision != Precision.float:
        source_model = torch.jit.trace(model.to("cpu"), make_torch_inputs(input_spec))
        print(f"Quantizing model {model_name}.")
        onnx_compile_job = hub.submit_compile_job(
            model=source_model,
            input_specs=input_spec,
            device=hub_device,
            name=model_name,
            options="--target_runtime onnx",
        )

        if not precision.activations_type or not precision.weights_type:
            raise ValueError("Quantization is only supported if both weights and activations are quantized.")

        if calib_data_dir:
            dataset = LocalImageCalibDataset(calib_data_dir)
        elif calib_hf_dataset:
            dataset = HFCalibDataset(calib_hf_dataset)
        else:
            dataset = quantization_utils.get_calibration_data(model, input_spec, num_calibration_samples)

        calib_loader = DataLoader(dataset, batch_size=1)
        # input_key = input_spec.get_input_names()[0]
        input_keys = list(input_spec.keys())
        calib_tensors_dict = {key: [] for key in input_keys}

        for batch in itertools.islice(iter(calib_loader), num_calibration_samples):
            if isinstance(batch, dict):
                for k in input_keys:
                    calib_tensors_dict[k].append(batch[k])
            elif isinstance(batch, (list, tuple)) and len(input_keys) > 1:
                for k, img in zip(input_keys, batch):
                    calib_tensors_dict[k].append(img)
            else:
                k = input_keys[0]
                img = batch if not isinstance(batch, (list, tuple)) else batch[0]
                calib_tensors_dict[k].append(img)

        for key in calib_tensors_dict.keys():
            print(f"Calibration data for {key}: {len(calib_tensors_dict[key])} samples")
        calibration_data = calib_tensors_dict

        quantize_job = hub.submit_quantize_job(
            model=onnx_compile_job.get_target_model(),
            calibration_data=calibration_data,
            activations_dtype=precision.activations_type,
            weights_dtype=precision.weights_type,
            name=model_name,
            options=model.get_hub_quantize_options(precision),
        )
        if skip_compiling:
            return ExportResult(quantize_job=quantize_job)

    source_model = quantize_job.get_target_model() if quantize_job else torch.jit.trace(model.to("cpu"), make_torch_inputs(input_spec))
    model_compile_options = model.get_hub_compile_options(target_runtime, precision, compile_options, hub_device)
    print(f"Optimizing model {model_name} to run on-device")
    compile_job = cast(hub.client.CompileJob, hub.submit_compile_job(
        model=source_model,
        input_specs=input_spec,
        device=hub_device,
        name=model_name,
        options=model_compile_options,
    ))

    profile_job: Optional[hub.client.ProfileJob] = None
    if not skip_profiling:
        profile_options_all = model.get_hub_profile_options(target_runtime, profile_options)
        print(f"Profiling model {model_name} on a hosted device.")
        profile_job = cast(hub.client.ProfileJob, hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=hub_device,
            name=model_name,
            options=profile_options_all,
        ))

    inference_job: Optional[hub.client.InferenceJob] = None
    if not skip_inferencing:
        profile_options_all = model.get_hub_profile_options(target_runtime, profile_options)
        print(f"Running inference for {model_name} on a hosted device with example inputs.")
        sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=use_channel_last_format)
        inference_job = cast(hub.client.InferenceJob, hub.submit_inference_job(
            model=compile_job.get_target_model(),
            inputs=sample_inputs,
            device=hub_device,
            name=model_name,
            options=profile_options_all,
        ))

    if not skip_downloading:
        os.makedirs(output_path, exist_ok=True)
        target_model = compile_job.get_target_model()
        assert target_model is not None
        target_model.download(str(output_path / model_name))

    if not skip_summary and profile_job is not None:
        assert profile_job.wait().success, "Job failed: " + profile_job.url
        profile_data: dict[str, Any] = profile_job.download_profile()
        print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and inference_job is not None:
        sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)
        torch_out = torch_inference(model, sample_inputs, return_channel_last_output=use_channel_last_format)
        assert inference_job.wait().success, "Job failed: " + inference_job.url
        inference_result = inference_job.download_output_data()
        assert inference_result is not None
        print_inference_metrics(inference_job, inference_result, torch_out, model.get_output_names())

    if not skip_summary:
        print_on_target_demo_cmd(compile_job, Path(__file__).parent, hub_device)

    return ExportResult(
        compile_job=compile_job,
        inference_job=inference_job,
        profile_job=profile_job,
        quantize_job=quantize_job,
    )

def main():
    warnings.filterwarnings("ignore")
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a8: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a16: [
            TargetRuntime.QNN,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
    }

    parser = export_parser(model_cls=Model, supported_precision_runtimes=supported_precision_runtimes)
    parser.add_argument("--calib_data_dir", type=str, default=None, help="Path to calibration image folder")
    parser.add_argument("--calib_hf_dataset", type=str, default=None, help="Hugging Face dataset name for calibration")
    args = parser.parse_args()
    validate_precision_runtime(supported_precision_runtimes, args.precision, args.target_runtime)
    export_model(**vars(args))

if __name__ == "__main__":
    main()
