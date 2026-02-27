from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch, os

# ABI
abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={torch._C._GLIBCXX_USE_CXX11_ABI}"

# 架构列表，默认覆盖 TITAN RTX (7.5) + 安培/霍普 (8.x/9.x)
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9")

extra_cxx_args = [
    "-O0",
    "-std=c++17",
    abi_flag,
]

extra_nvcc_args = [
    "-O0",
    "-std=c++17",
    abi_flag,
    "--expt-relaxed-constexpr",
    "-Xcompiler", "-fPIC",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

# 添加 gencode
for arch in arch_list.split(";"):
    if "." in arch:
        major, minor = arch.split(".")
    else:
        major, minor = arch, "0"
    code = f"sm_{major}{minor}"
    extra_nvcc_args += [f"-gencode=arch=compute_{major}{minor},code={code}"]

# 调试模式
if os.environ.get("DEBUG", "0") == "1":
    extra_cxx_args.append("-g")
    extra_nvcc_args.append("-G")

setup(
    name="HAIS_OP",
    ext_modules=[
        CUDAExtension(
            name="HAIS_OP",
            sources=[
                "src/hais_ops_api.cpp",
                "src/hais_ops.cpp",
                "src/cuda.cu",
            ],
            include_dirs=[
                os.path.join(torch.__path__[0], "include"),
                os.path.join(torch.__path__[0], "include", "torch", "csrc", "api", "include"),
            ],
            extra_compile_args={
                "cxx": extra_cxx_args,
                "nvcc": extra_nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
