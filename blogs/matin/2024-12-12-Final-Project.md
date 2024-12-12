---
author: matinraayai
format:
 html: default
title: Final Project: Lowering AMDGPU LLVM Intrinsics In Instrumentation Functions
---

For the past year or so I've been working on a framework to instrument AMD GPU code objects at runtime, similar to [NVBit](https://github.com/NVlabs/NVBit) for NVIDIA GPUs and [GTPin](https://www.intel.com/content/www/us/en/developer/articles/tool/gtpin.html) for Intel GPUs. My final project is essentially a major feature that I wanted to implement in the framework. I think it will be helpful to briefly go over the framework and how it is designed to better understand my final project and the challenges faced when
implementing it.
## Background
Instrumentation, in a nutshell, entails the modification of a binary in some shape or form, with the goal of gaining a better understanding of how it works. The motivation behind instrumentation ranges from debugging and profiling (e.g. [Valgrind](https://valgrind.org/), [Sanatizers](https://github.com/google/sanitizers/wiki/AddressSanitizer)), to architectural research (e.g. recording the addresses accessed by a target workload to better design future caches).
Instrumentation has been done for quite some time on the CPU side; Some examples include Intel's [Pin](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-dynamic-binary-instrumentation-tool.html), [DynInst](https://github.com/dyninst/dyninst), and [DynamoRio](https://dynamorio.org/). In recent years, this capability has been extended to NVIDIA GPUs via
[NVBit](https://github.com/NVlabs/NVBit) and to Intel GPUs via
[GTPin](https://www.intel.com/content/www/us/en/developer/articles/tool/gtpin.html). Both these frameworks are capable of "dynamic" instrumentation, which means they don't require access to the source code and modify the binary directly, just before the code runs.
"Static" instrumentation, on the other hand, instruments the binary "offline", and usually during compile time as part of the compilation process. Generally, dynamic instrumentation is preferred since it doesn't require the recompilation of the target code from scratch, which is impossible to do for closed-source applications. Another advantage of dynamic frameworks is the ability to switch between instrumented and original versions of the binary (also referred to as "selective instrumentation"), as they have access to both versions. This helps tremendously with reducing the overhead caused by instrumentation, which often times is very significant. Static instrumentation, on the other hand, can generated more efficient instrumented code as they have access to the compilation steps and analysis of the target code otherwise not available to dynamic framworks. They also don't incur a code-generation penalty compared to dynamic frameworks, as this step happens offline in static frameworks.

So far there hasn't been a successful attempt at creating a framework for dynamically instrumenting AMD GPU applications; Luthier (the framework that I've been working on as part of my PhD research), to the best of my (and my collegeues') knowledege, will be the first ever dynamic instrumentation framework targeting AMD GPUs. In the next section I go over the challenges I faced when designing Luthier, and then go over how it works and how it interfaces with AMD's hardware and software stack. Since this is a compiler class I will mostly focus on the compiler aspects of Luthier and I only briefly mention details on interfacing with the rest of the [ROCm](https://www.amd.com/en/products/software/rocm.html) stack. I will make these details available in the near future.

## Luthier: How It Was Designed, and How It Works
The initial design of Luthier was heavily inspired by NVBit. NVBit tools are `.so` shared objects that are loaded before the CUDA application using the [`LD_PRELOAD` trick](https://www.baeldung.com/linux/ld_preload-trick-what-is). The tools use [CUPTI](https://docs.nvidia.com/cupti/index.html) APIs to get notified when the CUDA application hit certain "events" (e.g. calls `cudaMalloc`, launches a kernel, etc.) through invoking a "callback" function defined inside the tool. NVBit tools use these callbacks to intercept CUDA kernels before they are launched, and afterwards inspect them using [`nvdisasm`](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html). Luthier tools works similiar to NVBit tools: they are shared objects loaded using the `LD_PRELOAD` environment variable; To be able to intercept and inspect kernels we designed a similar mechanism using [`rocprofiler-sdk`](https://github.com/ROCm/rocprofiler-sdk/) to notify tools about key events in the application. We then use the ROCr (HSA) runtime APIs to locate where the kernel has been loaded on the target GPU device, and even the ELF (i.e. code object) this kernel was loaded from.

But this was where we ran into the following issues:
1. **We realized that instrumenting AMDGPU code objects the "NVBit way" is not feasible or at best, very hard to implement**: NVBit instruments GPU applications by replacing the target instruction with a `JMP` to a trampoline region. The trampoline then spills the application's register onto the thread's stack, sets up the arguments to a instrumentation device function call, and the proceeds to call it. After returning, the trampoline will restore the registers, execute the original instruction, and then jumps back to the original kernel. This design is successful because SASS (NVIDIA GPU's hardware assembly) instructions have a fixed size, meaning that it's very easy to replace a single instruction with a "long jump" to almost any virtual memory address on the device, all without changing the layout of the code. Not changing the code layout is very important in dynamic instrumentation, as it ensures indirect jumps/calls won't break. This is not the case on AMD GPUs, as CDNA GPUs have a 4-byte short jump, only covering $2^{18}$ bytes, and an 8-byte long jump. To make matters worse, the long jump requires additional instructions to load the jump target into registers. We can argue that we might be able to make the NVBit trampoline work by allocating fixed-address executable memory using a custom allocator (which NVBit seem to also have),  but this goes completely against ROCr's APIs of asking for executable regions of memory using the `hsa_executable_t` interfaces, and is very hard to implement and manage. Even then this is only a temporary fix, as the aggregation of trampoline logic for each instruction will quickly go over the range managable by the short jump instruction. This meant we needed a completely different approach for instrumentation.
2.

### Inspecting Code Objects


### Generating Instrumentation Code

### Step 1. Intercepting AMD GPU Code Objects

### Step 2. Analyzing the

## New Feature: Lowering AMDGPU
On NVIDIA GPUs, accessing this property is very easy; There are dedicated registers the hardware can query to obtain these values; On AMD GPUs, however, there are no dedicated registers for these; Instead, they are either passed as arguments to the SGPRs or

First we go over the implementation of Luthier and then we explain how


One capability I noticed
