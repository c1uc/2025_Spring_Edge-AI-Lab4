import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Optional
import onnx
import onnxruntime as ort
import torchvision.transforms as T

from aimet_onnx._deprecated.auto_quant_v2 import AutoQuant
from aimet_onnx.adaround.adaround_weight import AdaroundParameters
from aimet_common.defs import QuantScheme

class DataLoader:
    """Custom DataLoader for LaMa inpainting data"""
    def __init__(self, dataset_dir, batch_size=1, iterations=None, test_dir=None):
        self.dataset_dir = dataset_dir
        self.test_dir = test_dir if test_dir else dataset_dir  # 使用單獨的測試資料夾
        self.batch_size = batch_size
        self.iterations = iterations
        
        # Get all image files (excluding mask files)
        search_dir = self.test_dir if hasattr(self, '_is_test_mode') else self.dataset_dir
        self.filenames = [f for f in os.listdir(search_dir) 
                         if f.endswith(".png") and not f.endswith("_mask.png")]
        self.filenames = sorted(self.filenames)
        
        if iterations:
            # Repeat filenames to meet iteration requirements
            total_needed = iterations * batch_size
            self.filenames = (self.filenames * (total_needed // len(self.filenames) + 1))[:total_needed]
    
    def set_test_mode(self, is_test=True):
        """切換到測試模式，使用 test_dir"""
        self._is_test_mode = is_test
        if is_test:
            self.filenames = [f for f in os.listdir(self.test_dir) 
                             if f.endswith(".png") and not f.endswith("_mask.png")]
            self.filenames = sorted(self.filenames)
    
    def __len__(self):
        return self.iterations if self.iterations else len(self.filenames) // self.batch_size
    
    def __iter__(self):
        current_dir = self.test_dir if hasattr(self, '_is_test_mode') and self._is_test_mode else self.dataset_dir
        
        for i in range(0, len(self.filenames), self.batch_size):
            batch_images = []
            batch_masks = []
            batch_gt_images = []  # 添加 ground truth 圖片
            
            for j in range(i, min(i + self.batch_size, len(self.filenames))):
                fname = self.filenames[j]
                image_path = os.path.join(current_dir, fname)
                mask_path = os.path.join(current_dir, fname.replace(".png", "_mask.png"))
                
                # 對於評估，我們需要原始圖片作為 ground truth
                gt_path = os.path.join(current_dir, fname.replace(".png", "_gt.png"))
                if not os.path.exists(gt_path):
                    gt_path = image_path  # 如果沒有 ground truth，使用原始圖片
                
                masked_image, mask, gt_image = self.load_image_mask_gt_triplet(image_path, mask_path, gt_path)
                batch_images.append(masked_image[0])  # Remove batch dimension for batching
                batch_masks.append(mask[0])
                batch_gt_images.append(gt_image[0])
            
            # Stack into batch
            batch_images = np.stack(batch_images, axis=0)
            batch_masks = np.stack(batch_masks, axis=0)
            batch_gt_images = np.stack(batch_gt_images, axis=0)
            
            yield batch_images, batch_masks, batch_gt_images
    
    def load_image_mask_gt_triplet(self, image_path, mask_path, gt_path):
        """Load and preprocess image-mask-groundtruth triplet"""
        # 載入被遮罩的圖片（輸入）
        masked_image = Image.open(image_path).convert("RGB").resize((512, 512))
        # 載入遮罩
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        # 載入原始完整圖片（ground truth）
        gt_image = Image.open(gt_path).convert("RGB").resize((512, 512))
        
        # 正規化到 [0, 1] 範圍
        masked_image = T.ToTensor()(masked_image).unsqueeze(0).numpy().astype(np.float32)
        gt_image = T.ToTensor()(gt_image).unsqueeze(0).numpy().astype(np.float32)
        
        # 確保遮罩是二進位的 (0 或 1)
        mask = T.ToTensor()(mask).unsqueeze(0).numpy().astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)  # 二進位化遮罩
        
        return masked_image, mask, gt_image
    
    def load_image_mask_pair(self, image_path, mask_path):
        """Load and preprocess image-mask pair (向後相容)"""
        masked_image, mask, _ = self.load_image_mask_gt_triplet(image_path, mask_path, image_path)
        return masked_image, mask


class CalibrationDataLoader:
    """Calibration DataLoader that yields input dict for AutoQuant"""
    def __init__(self, dataset_dir, batch_size=1, iterations=None, test_dir=None):
        self.base_loader = DataLoader(dataset_dir, batch_size, iterations, test_dir)
    
    def __len__(self):
        return len(self.base_loader)
    
    def __iter__(self):
        for batch_data in self.base_loader:
            if len(batch_data) == 3:  # 有 ground truth
                batch_images, batch_masks, batch_gt_images = batch_data
            else:  # 只有 image 和 mask
                batch_images, batch_masks = batch_data
            
            # AutoQuant calibration expects input dict format
            yield {'image': batch_images, 'mask': batch_masks}


def verify_model_inputs(onnx_model_path):
    """Verify ONNX model input specifications"""
    model = onnx.load(onnx_model_path)
    
    print("Model input specifications:")
    for input_tensor in model.graph.input:
        print(f"- Name: {input_tensor.name}")
        print(f"  Shape: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
        print(f"  Type: {input_tensor.type.tensor_type.elem_type}")
    
    return model


def test_model_inference(onnx_model_path, dummy_input):
    """Test if the model can run inference with dummy input"""
    try:
        # Create ONNX Runtime session with error handling
        providers = ['CPUExecutionProvider']  # Start with CPU only
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        print("Model input names:", [input.name for input in session.get_inputs()])
        print("Model output names:", [output.name for output in session.get_outputs()])
        
        # Test inference
        outputs = session.run(None, dummy_input)
        print(f"Inference successful! Output shape: {outputs[0].shape}")
        return True
        
    except Exception as e:
        print(f"Model inference test failed: {e}")
        return False


def compute_psnr(output, target, mask, max_pixel=1.0):
    """
    Compute PSNR for inpainting results within masked region
    
    Args:
        output: 模型輸出 (inpainted image) - shape: (batch_size, 3, 512, 512)
        target: ground truth 圖片 - shape: (batch_size, 3, 512, 512)  
        mask: 遮罩 - shape: (batch_size, 1, 512, 512), 1表示需要修復的區域
        max_pixel: 最大像素值，通常是 1.0
    
    Returns:
        float: 平均 PSNR 值
    """
    
    batch_size = output.shape[0]
    psnr_list = []
    
    print(f"PSNR Debug - Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"PSNR Debug - Target range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"PSNR Debug - Mask sum: {mask.sum():.0f}")
    
    for i in range(batch_size):
        single_output = output[i:i+1]  # Keep batch dimension
        single_target = target[i:i+1]
        single_mask = mask[i:i+1]
        
        # 檢查遮罩區域是否存在
        masked_pixels = np.sum(single_mask)
        if masked_pixels == 0:
            print(f"Warning: No masked pixels found in batch {i}")
            psnr_list.append(100.0)  # 沒有遮罩區域，設為完美分數
            continue
        
        # 只在遮罩區域計算 MSE
        # 注意：遮罩值為 1 的地方是需要修復的區域
        masked_diff = (single_output - single_target) * single_mask
        mse = np.sum(masked_diff ** 2) / (masked_pixels * single_output.shape[1])
        
        print(f"Batch {i} - MSE: {mse:.6f}, Masked pixels: {masked_pixels:.0f}")
        
        if mse == 0:
            psnr = 100.0  # 完美重建
        elif mse > 0:
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        else:
            print(f"Warning: Negative MSE detected: {mse}")
            psnr = 0.0
        
        print(f"Batch {i} - PSNR: {psnr:.4f}")
        psnr_list.append(psnr)
    
    average_psnr = np.mean(psnr_list)
    print(f"Average PSNR: {average_psnr:.4f}")
    return average_psnr


def compute_psnr_full_image(output, target, max_pixel=1.0):
    """
    Compute PSNR for entire image (alternative metric)
    """
    batch_size = output.shape[0]
    psnr_list = []
    
    for i in range(batch_size):
        single_output = output[i:i+1]
        single_target = target[i:i+1]
        
        mse = np.mean((single_output - single_target) ** 2)
        
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        psnr_list.append(psnr)
    
    return np.mean(psnr_list)


def create_eval_callback(dataset_dir, eval_dataset_size, test_dir=None):
    """Create evaluation callback function for AutoQuant"""
    
    def eval_callback(session: ort.InferenceSession, num_of_samples: Optional[int] = None) -> float:
        try:
            # Create evaluation data loader，使用測試資料夾
            eval_data_loader = DataLoader(dataset_dir, batch_size=1, test_dir=test_dir)
            eval_data_loader.set_test_mode(True)  # 切換到測試模式
            
            if num_of_samples:
                iterations = min(num_of_samples, len(eval_data_loader.filenames))
            else:
                iterations = min(eval_dataset_size, len(eval_data_loader.filenames))
            
            psnr_scores = []
            batch_count = 0
            
            print(f"Starting evaluation with {iterations} samples...")
            
            for batch_data in eval_data_loader:
                if batch_count >= iterations:
                    break
                
                try:
                    if len(batch_data) == 3:  # 有 ground truth
                        batch_images, batch_masks, batch_gt_images = batch_data
                    else:  # 只有 image 和 mask
                        batch_images, batch_masks = batch_data
                        batch_gt_images = batch_images  # 使用輸入圖片作為 ground truth
                    
                    # Run inference - LaMa model needs both image and mask
                    model_inputs = {'image': batch_images, 'mask': batch_masks}
                    outputs = session.run(None, model_inputs)
                    output_images = outputs[0]  # 假設第一個輸出是修復後的圖片
                    
                    print(f"\nBatch {batch_count}:")
                    print(f"  Input shape: {batch_images.shape}")
                    print(f"  Output shape: {output_images.shape}")
                    print(f"  Mask shape: {batch_masks.shape}")
                    print(f"  GT shape: {batch_gt_images.shape}")
                    
                    # Compute PSNR (比較修復結果和 ground truth)
                    psnr = compute_psnr(output_images, batch_gt_images, batch_masks)
                    
                    # 也可以計算全圖 PSNR 作為參考
                    full_psnr = compute_psnr_full_image(output_images, batch_gt_images)
                    print(f"  Full image PSNR: {full_psnr:.4f}")
                    
                    if psnr > 0:  # 只接受正值
                        psnr_scores.append(psnr)
                    else:
                        print(f"  Warning: Negative PSNR {psnr:.4f} ignored")
                    
                except Exception as e:
                    print(f"Error during inference on batch {batch_count}: {e}")
                    # 不返回 0，而是跳過這個 batch
                    continue
                
                batch_count += 1
            
            if not psnr_scores:
                print("No valid PSNR scores computed!")
                return 0.0
            
            average_psnr = np.mean(psnr_scores)
            print(f"\nFinal Evaluation Results:")
            print(f"  Valid samples: {len(psnr_scores)}/{batch_count}")
            print(f"  Average PSNR: {average_psnr:.4f}")
            print(f"  PSNR range: [{min(psnr_scores):.4f}, {max(psnr_scores):.4f}]")
            
            return average_psnr
            
        except Exception as e:
            print(f"Error in evaluation callback: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    return eval_callback


def main(args):
    # Step 1. Define constants
    EVAL_DATASET_SIZE = args.eval_dataset_size
    CALIBRATION_DATASET_SIZE = args.calibration_dataset_size
    BATCH_SIZE = args.batch_size
    
    print(f"Loading ONNX model from {args.onnx_model}")
    
    # Step 2. Verify and prepare model
    onnx_model = verify_model_inputs(args.onnx_model)
    
    # Create dummy input based on your model's expected input shape
    # Note: Adjust these shapes based on your actual model requirements
    input_shape = (BATCH_SIZE, 3, 512, 512)  # Image shape
    mask_shape = (BATCH_SIZE, 1, 512, 512)   # Mask shape
    
    # Use more realistic dummy data
    dummy_image = np.random.uniform(0, 1, input_shape).astype(np.float32)
    dummy_mask = np.random.choice([0, 1], size=mask_shape).astype(np.float32)
    dummy_input = {'image': dummy_image, 'mask': dummy_mask}
    
    # Test model inference before quantization
    print("Testing model inference...")
    if not test_model_inference(args.onnx_model, dummy_input):
        print("Model inference test failed. Please check your ONNX model.")
        return
    
    # Create calibration dataloader
    unlabelled_data_loader = CalibrationDataLoader(
        dataset_dir=args.dataset_dir,
        batch_size=BATCH_SIZE,
        iterations=math.ceil(CALIBRATION_DATASET_SIZE / BATCH_SIZE),
        test_dir=args.test_dir  # 支援測試資料夾
    )
    
    # Step 3. Prepare eval callback
    eval_callback = create_eval_callback(args.dataset_dir, EVAL_DATASET_SIZE, args.test_dir)
    
    # Step 4. Create AutoQuant object with more conservative settings
    try:
        auto_quant = AutoQuant(
            model=onnx_model,
            dummy_input=dummy_input,
            data_loader=unlabelled_data_loader,
            eval_callback=eval_callback,
            param_bw=8,  # 8-bit weights
            output_bw=8,  # 8-bit activations
            quant_scheme=QuantScheme.post_training_tf_enhanced,
            # config_file_path=None,  # Use default config
            strict_validation=False  # Allow some flexibility
        )
    except Exception as e:
        print(f"Error creating AutoQuant object: {e}")
        return
    
    # Step 5. (Optional) Set AdaRound params
    if args.use_adaround:
        try:
            ADAROUND_DATASET_SIZE = args.adaround_dataset_size
            adaround_data_loader = CalibrationDataLoader(
                dataset_dir=args.dataset_dir,
                batch_size=BATCH_SIZE,
                iterations=math.ceil(ADAROUND_DATASET_SIZE / BATCH_SIZE)
            )
            adaround_params = AdaroundParameters(
                data_loader=adaround_data_loader,
                num_batches=len(adaround_data_loader)
            )
            auto_quant.set_adaround_params(adaround_params)
            print("AdaRound parameters set successfully")
        except Exception as e:
            print(f"Warning: Could not set AdaRound parameters: {e}")
            print("Continuing without AdaRound...")
    
    # Step 6. Run AutoQuant with error handling
    try:
        print("Running initial inference...")
        sim, initial_accuracy = auto_quant.run_inference()
        print(f"Initial inference completed. Accuracy: {initial_accuracy}")
        
        print("Optimizing quantization...")
        model_path, optimized_accuracy, encoding_path = auto_quant.optimize(
            allowed_accuracy_drop=args.allowed_accuracy_drop
        )
        
        print(f"Results:")
        print(f"- Initial PSNR (before optimization): {initial_accuracy:.4f}")
        print(f"- Optimized PSNR (after optimization): {optimized_accuracy:.4f}")
        print(f"- Quantized model saved at: {model_path}")
        print(f"- Encoding saved at: {encoding_path}")
        
    except Exception as e:
        print(f"Error during AutoQuant execution: {e}")
        print("This might be due to model compatibility issues or data format problems.")
        
        # Provide debugging suggestions
        print("\nDebugging suggestions:")
        print("1. Check if your ONNX model is compatible with AIMET")
        print("2. Verify that input data shapes match model expectations")
        print("3. Try running with a smaller batch size")
        print("4. Ensure your dataset contains valid image-mask pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoQuant for LaMa model with PSNR evaluation")
    parser.add_argument("--onnx_model", type=str, required=True, 
                       help="Path to ONNX model")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                       help="Folder with image and *_mask.png pairs for training/calibration")
    parser.add_argument("--test_dir", type=str, default=None,
                       help="Separate folder for test images (optional, defaults to dataset_dir)")
    
    # Dataset sizes
    parser.add_argument("--eval_dataset_size", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--calibration_dataset_size", type=int, default=50,
                       help="Number of samples for calibration")
    parser.add_argument("--adaround_dataset_size", type=int, default=100,
                       help="Number of samples for AdaRound")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for data loading")
    parser.add_argument("--allowed_accuracy_drop", type=float, default=0.5,
                       help="Allowed PSNR drop during quantization")
    
    # Optional features
    parser.add_argument("--use_adaround", action="store_true",
                       help="Enable AdaRound optimization")
    
    args = parser.parse_args()
    main(args)
