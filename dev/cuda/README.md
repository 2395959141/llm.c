# dev/cuda

此目录是用于开发各种所需 CUDA 内核的工作空间。每个文件都开发一个内核，通常包含该内核的多个版本，这些版本可能具有不同的运行时间和不同的代码或时间复杂度。

有关如何编译和运行内核的说明，请参见每个文件的顶部。或者，为了方便起见，所有命令也都已分组在此目录的 `Makefile` 中。

例如，我们可以查看 `layernorm_forward.cu` 的顶部，以构建 LayerNorm 的前向传播内核：

```bash
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward
```

或者简单地使用：

```bash
make layernorm_forward
```

文件顶部的注释记录了此内核的不同版本，通常这些版本的复杂度逐渐增加，运行时间逐渐减少。例如，查看文件顶部的注释，我们可以运行最简单的内核：

```bash
./layernorm_forward 1
```

你会看到，这首先在 CPU 上运行参考代码，然后在 GPU 上运行内核 1，比较结果以检查正确性，然后运行多个配置的内核（最常见和最重要的是块大小），以在这些启动配置中计时内核。然后我们可以运行更快的内核之一（内核 4）：

```bash
./layernorm_forward 4
```

你会看到这与所有 CPU 结果都匹配，但运行速度快得多。从这里开始的典型过程是，我们复制粘贴运行最快的内核，手动调整它（例如，硬编码最佳块大小）并将其放入训练代码文件中，例如 `train_gpt2.cu`。

要添加内核的新版本，请将内核添加到相应文件并调整文档。要添加新内核，请添加新文件并调整 Makefile。运行 `make clean` 以清理目录中的二进制文件。

如果你没有 GPU 或遇到 CUDA 依赖问题，可以在 [Modal 平台](http://modal.com) 上运行基准测试。例如，要在具有 80GB 内存的 A100 GPU 上运行注意力前向传播的基准测试，可以运行以下命令：

```bash
GPU_MEM=80 modal run benchmark_on_modal.py --compile-command "nvcc -O3 --use_fast_math attention_forward.cu -o attention_forward -lcublas" --run-command "./attention_forward 1"
```
