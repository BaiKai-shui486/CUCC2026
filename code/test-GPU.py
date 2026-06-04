"""
GPU 测试脚本 - 检查系统是否能正确识别和使用 GPU

功能:
1. 检测可用的计算设备 (CPU/GPU/MPS)
2. 显示 GPU 详细信息 (如果可用)
3. 执行简单的 GPU 计算测试
4. 验证 PyTorch CUDA 支持
"""

import torch
import platform
import sys


def print_separator(title=""):
    """打印分隔线"""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def check_system_info():
    """检查系统基本信息"""
    print_separator("系统信息")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 编译版本: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")


def check_cuda_availability():
    """检查 CUDA 可用性"""
    print_separator("CUDA 可用性检查")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n--- GPU {i} ---")
            print(f"  设备名称: {torch.cuda.get_device_name(i)}")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
            
            # 获取内存信息
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_memory, _ = torch.cuda.mem_get_info(i)
            used_memory = total_memory - free_memory
            
            print(f"  总显存: {total_memory / 1024**3:.2f} GB")
            print(f"  已用显存: {used_memory / 1024**3:.2f} GB")
            print(f"  空闲显存: {free_memory / 1024**3:.2f} GB")
    else:
        print("CUDA 不可用，将使用 CPU 进行计算")
    
    return cuda_available


def check_mps_availability():
    """检查 MPS (Metal Performance Shaders) 可用性 - 适用于 macOS"""
    print_separator("MPS 可用性检查 (macOS)")
    
    mps_available = torch.backends.mps.is_available()
    print(f"MPS 是否可用: {mps_available}")
    
    if mps_available:
        print("MPS 可用，可以使用 Apple Silicon GPU")
    else:
        print("MPS 不可用")
    
    return mps_available


def determine_device():
    """确定要使用的设备"""
    print_separator("设备选择")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("使用 CPU (未检测到 GPU)")
    
    print(f"当前设备: {device}")
    return device


def test_gpu_computation(device):
    """测试 GPU 计算能力"""
    print_separator("GPU 计算测试")
    
    try:
        # 创建大型张量进行测试
        size = 10000
        
        print(f"在 {device} 上创建大小为 {size}x{size} 的随机矩阵...")
        
        # 根据设备类型创建张量
        if device.type == 'cuda':
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
        elif device.type == 'mps':
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
        else:
            a = torch.randn(size, size)
            b = torch.randn(size, size)
        
        print("执行矩阵乘法测试...")
        
        import time
        start_time = time.time()
        
        # 执行矩阵乘法
        c = torch.matmul(a, b)
        
        # 对于 GPU，需要同步以确保计算完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"  矩阵乘法完成!")
        print(f"  计算时间: {elapsed_time:.4f} 秒")
        print(f"  结果形状: {c.shape}")
        print(f"  结果均值: {c.mean().item():.6f}")
        print(f"  结果标准差: {c.std().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"GPU 计算测试失败: {e}")
        return False


def test_tensor_operations(device):
    """测试基本张量操作"""
    print_separator("基本张量操作测试")
    
    try:
        # 创建测试张量
        if device.type in ['cuda', 'mps']:
            x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        else:
            x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        print(f"原始张量: {x}")
        
        # 执行各种操作
        y = x * 2
        z = x + y
        w = torch.sin(x)
        
        print(f"x * 2 = {y}")
        print(f"x + y = {z}")
        print(f"sin(x) = {w}")
        print(f"张量均值: {x.mean().item():.4f}")
        print(f"张量最大值: {x.max().item():.4f}")
        
        print("基本张量操作测试通过")
        return True
        
    except Exception as e:
        print(f"基本张量操作测试失败: {e}")
        return False


def test_neural_network(device):
    """测试简单神经网络"""
    print_separator("神经网络测试")
    
    try:
        import torch.nn as nn
        
        # 创建简单网络
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 初始化模型并移到指定设备
        model = SimpleNet().to(device)
        print(f"模型已加载到设备: {next(model.parameters()).device}")
        
        # 创建测试数据
        batch_size = 32
        input_data = torch.randn(batch_size, 10, device=device)
        
        # 前向传播
        output = model(input_data)
        print(f"输入形状: {input_data.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出均值: {output.mean().item():.6f}")
        
        print("神经网络测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 神经网络测试失败: {e}")
        return False


def main():
    """主函数"""
    print("GPU 检测和测试工具")
    print("=" * 60)
    
    # 1. 检查系统信息
    check_system_info()
    
    # 2. 检查 CUDA 可用性
    cuda_available = check_cuda_availability()
    
    # 3. 检查 MPS 可用性 (macOS)
    mps_available = check_mps_availability()
    
    # 4. 确定使用设备
    device = determine_device()
    
    # 5. 执行计算测试
    computation_success = test_gpu_computation(device)
    
    # 6. 测试基本张量操作
    tensor_ops_success = test_tensor_operations(device)
    
    # 7. 测试神经网络
    nn_success = test_neural_network(device)
    
    # 8. 总结
    print_separator("测试总结")
    
    if device.type == 'cuda':
        print(f"   GPU (CUDA) 检测成功!")
        print(f"   设备: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
    elif device.type == 'mps':
        print("MPS (Apple Silicon GPU) 检测成功!")
    else:
        print("仅检测到 CPU，未找到可用的 GPU")
    
    print(f"\n测试结果:")
    print(f"  矩阵计算测试: {'通过' if computation_success else '失败'}")
    print(f"  张量操作测试: {'通过' if tensor_ops_success else '失败'}")
    print(f"  神经网络测试: {'通过' if nn_success else '失败'}")
    
    all_passed = computation_success and tensor_ops_success and nn_success
    
    if all_passed:
        print(f"\n所有测试通过! 可以正常使用 {'GPU' if device.type != 'cpu' else 'CPU'} 进行深度学习计算。")
    else:
        print(f"\n部分测试失败，请检查上述错误信息。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)