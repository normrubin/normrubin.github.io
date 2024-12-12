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
1. **We realized that instrumenting AMDGPU code objects the "NVBit way" is not feasible or at best, very hard to implement**: NVBit instruments GPU applications by replacing the target instruction with a `JMP` to a trampoline region. The trampoline then spills the application's register onto the thread's stack, sets up the arguments to a instrumentation device function call, and the proceeds to call it. After returning, the trampoline will restore the registers, execute the original instruction, and then jumps back to the original kernel. This design is successful because SASS (NVIDIA GPU's hardware assembly) instructions have a fixed size, meaning that it's very easy to replace a single instruction with a "long jump" to almost any virtual memory address on the device, all without changing the layout of the code. Not changing the code layout is very important in dynamic instrumentation, as it ensures indirect jumps/calls won't break. This is not the case on AMD GPUs, as CDNA GPUs have a 4-byte short jump, only covering $2^{18}$ bytes, and an 8-byte long jump. To make matters worse, the long jump requires additional instructions to load the jump target into registers. We can argue that we might be able to make the NVBit trampoline work by allocating fixed-address executable memory using a custom allocator (which NVBit seem to also have),  but this goes completely against ROCr's APIs of asking for executable regions of memory using the `hsa_executable_t` interfaces, and is very hard to implement and manage. Even then this is only a temporary fix, as the aggregation of trampoline logic for each instruction will quickly go over the range managable by the short jump instruction. **This meant we needed a completely different approach for instrumentation**.
2. **Reusing Dead Registers**: NVBit inserts calls to pre-compiled instrumentation device functions with a set calling convention, which NVBit has no choice but to obey by spilling/restoring a large number of thread registers at all times. Accessing the stack is not cheap on GPUs, hence **we wanted to find a way to adapt the same instrumentation function to each instrumentation point to reuse dead registers** in order to speed up instrumented code.
3. **Allocating And Accessing Scratch (Local) Memory Is Not Trivial on AMD GPUs**: in the NVBit paper, accessing the stack (local memory) of each thread for spilling and restoring registers is mentioned; However, how this access is done without interfering with the application's local memory is not explained. We assume that NVBit assumes all SASS code adhere to NVIDIA's calling conventions, which is enforced via their virtual PTX ISA; Hence, it can always have a stack pointer which allows for interleaving the spilled registers with the application's local memory. AMD GPUs, however, can be programmed directly using hardware assembly, and have multiple calling conventions that don't enforce presence of a stack pointer register. **We needed to find a way to access the stack for instrumentation without the presnce of a stack pointer**.
4. **Requesting access to scratch might displace the SGPR arguments**: According to the [AMDGPU LLVM Docs](https://llvm.org/docs/AMDGPUUsage.html), when a kernel is launched, a set of wavefront-common values are loaded into SGPRs for use by the kernel. This includes the address of the kernel argument buffer or the address of the queue (i.e. command processor) used to launch the kernel. It also includes resources needed to access local/scratch memory in each thread. These resource must be explicitly requested by the kernel, and [this table](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-sgpr-register-set-up-order-table) shows the order these SGPRs are setup. One thing to note is that accessing scratch requires access to the "Private Segment Buffer Descriptor" and/or "Flat Scratch Address" of the queue used to launch the kernel, as well as a "wave offset", which is the offset from the queue's scratch address to the current wavefront's scratch space. This makes accessing scratch in instrumentation functions particularly challenging, especially for applications that don't require scratch and don't set it up. Luthier **must take into account these shift in SGPR arguments and must emit code to set it up for the instrumentation stack and then move the SGPR arguments to their original place**. It also have to **store the scratch information somewhere to be able to access it later in the instrumentation routines**.

Luthier addresses each of the afformentioned challenge as follows:
1. Instead of carrying out instrumentation by directly modifying the loaded code on the GPU, Luthier opts to create a "standalone executable", which contains the original code as well as the instrumentation logic. The standalone executable will use ROCr dynamic loader's features to link against the "static global variables" already loaded in the original code (e.g. variables annotated with `__device__` or `__managed__` in HIP code). This respects the ROCr APIs the most, not requiring any significant changes to the low level runtime.
2. Instead of calling pre-compiled device functions, Luthier embeds the optimized LLVM IR bitcode of the device instrumentation logic in its device code objects at tool compile time. This ensures that the HIP runtime will load the bitcode for free, allowing the Luthier runtime to do more optimizations on the instrumentation logic and adapt it to each instrumentation point.
3. (and 4.) Luthier defines a new concept called the "State Value Array (SVA)", an array of 64 32-bit values that can be loaded into a single VGPR of a single wavefront (each wavefront has 64 threads). SVA is in charge of storing the instrumentation stack information, which is always allocated on top of what the application requested originally. It also keeps track of other wavefront-specific values. SVA is setup using a "Kernel preamble code", which reads the SGPR arguments and saves them into the SVA's VGPR lanes, right before reverting the SGPR arguments back to their original formation before the target kernel starts executing. SVA is stored in the applications' dead/unused registers. In most recent hardware, the SVA can be either stored in a single A/VGPR, or spilled on a static point of the kernel stack with a single SGPR pointing to it so it can be loaded later using `SCRATCH` instructions.

Luthier implements these designs by heavily leveraging the LLVM project and its AMDGPU backend, which we explain in more detail below:

### Disassembling, Lifting, and Inspecting Code Objects
As we mentioned earlier, Luthier instruments code by duplicating the target application's code so that it can freely inject instrumentation logic inside it. To do this, Luthier takes in a single ELF/executable, and inspects its symbols using LLVM's object utilities. It then identifies the kernels and device functions inside the ELF and disassembles them into LLVM MC instructions. [MC](https://blog.llvm.org/2010/04/intro-to-llvm-mc-project.html) is the machine code/assembler layer of LLVM. It is meant to represent "physical" instructions and registers. While disassembling code, Luthier identifies the branch instructions and identifies their targets if possible for later use.

After disassembly is complete Luthier uses the obtained information from the ELF to "lift" it to LLVM Machine IR (MIR). [MIR](https://llvm.org/docs/CodeGenerator.html) is LLVM's representation used in its target-independent code generator (i.e. backends). It is a superset of LLVM MC, meaning an LLVM MC instruction can also be easily converted to
an LLVM MIR instruction by reusing the same enums for opcodes and registers. The reason behind lifting to LLVM MIR is as follows:
1. MIR has very convenient ways for iterating over the code and querying properties regarding the code and the instructions which is otherwise costly to implement ourselves.
2. MIR's high-level utilities allow for removing/adding things to the ELF otherwise not possible with ELF-modification frameworks e.g. [ELFIO](https://github.com/serge1/ELFIO/tree/main). This makes it easy to modify the kernel specifications and removing symbols that are not used in the target kernel and are unrelated.
3. The compiler machine passes only operate on the MIR representation. Lifting to MIR makes it easier to re-use analysis already available in LLVM's code generator or add new ones.

LLVM MIR consists of a set of "pseudo" instructions that is equivalent to a set of "physical", target-specific instructions in MC; For example, `S_ADD_U32` is a pseudo instruction in MIR which maps to `S_ADD_U32_vi`, `S_ADD_32_gfx12`, and so on. The same goes for some registers e.g. `TTMP` or `M0`. The primary reason for this indirection is to allow for different encoding/decodings on different targets. Although in theory, one should be able to use both pseudo and target-specific opcodes in the MIR, the AMDGPU backend primarily expects the pseudo variant to be present in the code; Hence during its tool library compilation, Luthier uses LLVM's [TableGen](https://llvm.org/docs/TableGen/) to read over all the opcodes/registers and creates an inverse mapping between target-specific opcode and their pseudo equivalents. Luthier then uses this table to convert the opcode and registers to their pseudo variants during runtime.

Just like [LLVM Bolt](https://github.com/llvm/llvm-project/blob/main/bolt/README.md), Luthier requires the inspected ELFs to have full relocation information. This way, it is able to correctly lift things like reading the address of a symbol:
```asm
s_getpc_b64 s[4:5]
s_add_u32 s4, s4, func1@rel32@lo+4
s_addc_u32 s5, s5, func1@rel32@lo+4
```

In the future, we will have to add analysis logic to at least reduce this restriction; For now, however, it is more than enough for a proof-of-concept. The result of the lifting process it a `LiftedRepresentation`, which is a mapping between the HSA/ROCr symbols of the ELF and their LLVM equivalent.


### Writing Luthier Tools
A sample Luthier tool can look like the following:

```c++

using namespace luthier;

/// Kernel instruction counter
__attribute__((managed)) uint64_t Counter = 0;

/// Macro marking the device module (code object) of this tool to
/// be a Luthier tool
MARK_LUTHIER_DEVICE_MODULE

LUTHIER_HOOK_ANNOTATE countInstructionsVector(bool CountWaveFrontLevel) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Get the position of the thread in the current wavefront (1-index)
  const uint32_t LaneId = __lane_id() + 1;
  // Get the first active thread id inside this wavefront
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  // Get the number of active threads in this wavefront
  uint32_t NumActiveThreads = __popcll(ExecMask);

  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == LaneId) {
    if (CountWaveFrontLevel) {
      // Num threads can be zero when accounting for predicates off
      if (NumActiveThreads > 0) {
        atomicAdd(&Counter, 1);
      }
    } else {
      atomicAdd(&Counter, NumActiveThreads);
    }
  }
}

LUTHIER_EXPORT_HOOK_HANDLE(countInstructionsVector);

LUTHIER_HOOK_ANNOTATE countInstructionsScalar() {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Overwrite the exec mask with one so that only a single thread is active
  luthier::writeExec(1);
  // Increment the counter by 1
  atomicAdd(&Counter, 1);
  // Restore the exec mask
  luthier::writeExec(ExecMask);
}

LUTHIER_EXPORT_HOOK_HANDLE(countInstructionsScalar);

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  // Create a constant bool indicating the CountWavefrontLevel value
  auto *CountWavefrontLevelConstVal =
      llvm::ConstantInt::getBool(LR.getContext(), CountWavefrontLevel);
  unsigned int I = 0;
  for (auto &[_, MF] : LR.functions()) {
    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        if (I >= InstrBeginInterval && I < InstrEndInterval) {
          bool IsScalar =
              llvm::SIInstrInfo::isSOP1(MI) || llvm::SIInstrInfo::isSOP2(MI) ||
              llvm::SIInstrInfo::isSOPK(MI) || llvm::SIInstrInfo::isSOPC(MI) ||
              llvm::SIInstrInfo::isSOPP(MI) || llvm::SIInstrInfo::isSMRD(MI);
          bool IsLaneAccess =
              MI.getOpcode() == llvm::AMDGPU::V_READFIRSTLANE_B32 ||
              MI.getOpcode() == llvm::AMDGPU::V_READLANE_B32 ||
              MI.getOpcode() == llvm::AMDGPU::V_WRITELANE_B32;
          if (IsScalar || IsLaneAccess)
            LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                MI, LUTHIER_GET_HOOK_HANDLE(countInstructionsScalar)));
          else
            LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                MI, LUTHIER_GET_HOOK_HANDLE(countInstructionsVector),
                {CountWavefrontLevelConstVal}));
        }
        I++;
      }
    }
  }
  return llvm::Error::success();
}

static void
instrumentAllFunctionsOfLR(const hsa::LoadedCodeObjectKernel &Kernel) {
  auto LR = lift(Kernel);
  LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());
  LUTHIER_REPORT_FATAL_ON_ERROR(
      instrumentAndLoad(Kernel, *LR, instrumentationLoop, "instr_count"));
}

/// ... Definition of Luthier HSA/ROCr callbacks goes here

```

Luthier tools are written in HIP/C++. This example tool first calls the `lift` function on the kernel of interest to obtain an instance of `LiftedRepresentation`. It then uses the `instrumentAndLoad` function to instrument the kernel and load it into ROCr. The `instrumentationLoop` is a lambda function that allows direct modification of the `LR` (the one inside the `instrumentAllFunctionsOfLR` is immutable) and population of an `InstrumentationTask` by calling the `insertHookBefore` function, letting the tool know that we want to insert a call to an instrumentation function (also called hooks).

Some special macros used in Luthier tools are as follows:
1. `MARK_LUTHIER_DEVICE_MODULE` has the following definition:
	```c++
	#define MARK_LUTHIER_DEVICE_MODULE \
	__attribute__((managed, used)) char __luthier_reserved = 0;
	```
	This macro ensures the device module of the Luthier tool is easily identifiable by the Luthier runtime. Also the managed variable ensures that our device module will be loaded right before the first HIP kernel launch. In some instances  where the target application directly uses the ROCr runtime, we enable eager loading in HIP using a special environment variable to ensure our tool device module is loaded right away.
2. The following macros are related to Luthier "hooks":
	```c++
	#define LUTHIER_HOOK_ANNOTATE \
  	__attribute__((device, used, annotate("luthier_hook"))) extern "C" void
	#define LUTHIER_EXPORT_HOOK_HANDLE(HookName) \
  	__attribute__((global, used)) extern "C" void __luthier_hook_handle_##HookName(){};
		#define LUTHIER_GET_HOOK_HANDLE(HookName)     \
  	reinterpret_cast<const void *>(__luthier_hook_handle_##HookName)
	```
	Hook is an special instrumentation function that can be called right before an instruction of a application. It will be inlined inside the call site while ensuring correct register and stack usage. A hook can also call other device functions and they don't have to be inlined themselves. Hooks can take arguments to Register values and LLVM `Constant`s (e.g. `countInstructionsVector` takes a bool argument which is setup inside `instrumentationLoop`).

	As device functions in HIP don't get a handle accessible from the host code, the `LUTHIER_EXPORT_HOOK_HANDLE` macro creates a host-accessible dummy handle that can be used by the host logic using  `LUTHIER_GET_HOOK_HANDLE`. The `__attribute__(used)` ensures the compiler doesn't optimize these symbols away, as they are needed for Luthier's correct functionality.

3. Luthier doesn't allow usage of inline assembly inside its hooks or any of its called device functions as their register usage cannot be analyzed until the very last step of code generation; Instead, it introduces a new concept called a "Luthier Intrinsic". Luthier intrinsics are meant to behave like LLVM intrinsics, as they end up translating to a sequence of low-level code. Luthier itself has a set of pre-implemented intrinsics, including `luthier::readReg` and `luthier::writeReg`, which read and write values to the registers. In this example, we use the `luthier::writeExec` intrinsic:
	```c++
	#define LUTHIER_INTRINSIC_ANNOTATE \
	__attribute__((device, noinline, annotate("luthier_intrinsic")))

	template <typename T>
	__attribute__((device, always_inline)) void doNotOptimize(T const &Value) {
		__asm__ __volatile__("" : : "X"(Value) : "memory");
	}

	LUTHIER_INTRINSIC_ANNOTATE void writeExec(uint64_t Val) {
	  doNotOptimize(Val);
	}
	```
	The `LUTHIER_INTRINSIC_ANNOTATE` macro annotates any Luthier intrinsic device functions used in hooks. Since HIP doesn't allow calls to `extern` device functions (due to lack of support for device functions in the ROCr loader), we need to emit a body for it and ensure the compiler doesn't optimize any of the arguments away and the intrinsic calls away. The `doNotOptimize` template device function is used for exactly this purpose, which is borrowed from [Google Benchmark Framework](https://stackoverflow.com/questions/66795357/google-benchmark-frameworks-donotoptimize).

As mentioned earlier, Luthier does not rely on inserting calls to a pre-compiled instrumentation device function. Luthier instead, embeds a pre-processed LLVM IR bitcode that it embeds inside the device code object. At tool compile time, Luthier utilizes a custom compiler plugin, implemented as follows:
```c++
//===-- EmbedInstrumentationModuleBitcodePass.cpp -------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the \c luthier::EmbedInstrumentationModuleBitcode pass,
/// used by Luthier tools to preprocess instrumentation modules and embedding
/// them inside device code objects.
//===----------------------------------------------------------------------===//

#include "EmbedInstrumentationModuleBitcodePass.hpp"

#include "llvm/Passes/PassPlugin.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-embed-optimized-bitcode-pass"

namespace luthier {

// TODO: Import these variables as well as the static functions from
//  Luthier proper once the separate compilation issue is resolved

static constexpr const char *ReservedManagedVar = "__luthier_reserved";

static constexpr const char *HookAttribute = "luthier_hook";

static constexpr const char *IntrinsicAttribute = "luthier_intrinsic";

static constexpr const char *HipCUIDPrefix = "__hip_cuid_";

/// Builds a \c llvm::CallInst invoking the intrinsic indicated by
/// \p IntrinsicName at the instruction position indicated by the \p Builder
/// with the given \p ReturnType and \p Args
/// \tparam IArgs Arguments passed to the intrinsic; Can be either a scalar
/// or a reference to a \c llvm::Value
/// \param M the instrumentation module where the intrinsic will be inserted to
/// \param Builder the instruction builder used to build the call instruction
/// \param IntrinsicName the name of the intrinsic
/// \param ReturnType the return type of the intrinsic call instruction
/// \param Args the arguments to the intrinsic function
/// \return a \c llvm::CallInst to the intrinsic function
llvm::CallInst *insertCallToIntrinsic(llvm::Module &M,
                                      llvm::IRBuilderBase &Builder,
                                      llvm::StringRef IntrinsicName,
                                      llvm::Type &ReturnType) {
  auto &LLVMContext = Builder.getContext();
  /// Construct the intrinsic's LLVM function type and its argument value
  /// list
  auto *IntrinsicFuncType = llvm::FunctionType::get(&ReturnType, false);
  // Format the readReg intrinsic function name
  std::string FormattedIntrinsicName{IntrinsicName};
  llvm::raw_string_ostream IntrinsicNameOS(FormattedIntrinsicName);
  // Format the intrinsic function name
  IntrinsicNameOS << ".";
  IntrinsicFuncType->getReturnType()->print(IntrinsicNameOS);
  // Create the intrinsic function in the module, or get it if it already
  // exists
  auto ReadRegFunc = M.getOrInsertFunction(
      FormattedIntrinsicName, IntrinsicFuncType,
      llvm::AttributeList().addFnAttribute(LLVMContext, IntrinsicAttribute,
                                           IntrinsicName));

  return Builder.CreateCall(ReadRegFunc);
}

/// Given a function's mangled name \p MangledFuncName,
/// partially demangles it and returns the base function name with its
/// namespace prefix \n
/// For example given a demangled function name int a::b::c<int>(), this
/// method returns a::b::c
/// \param MangledFuncName the mangled function name
/// \return the name of the function with its namespace prefix
static std::string
getDemangledFunctionNameWithNamespace(llvm::StringRef MangledFuncName) {
  // Get the name of the function, without its template arguments
  llvm::ItaniumPartialDemangler Demangler;
  // Ensure successful partial demangle operation
  if (Demangler.partialDemangle(MangledFuncName.data()))
    llvm::report_fatal_error("Failed to demangle the intrinsic name " +
                             MangledFuncName + ".");
  // Output string
  std::string Out;
  // Output string's ostream
  llvm::raw_string_ostream OS(Out);

  size_t BufferSize;
  char *FuncNamespaceBegin =
      Demangler.getFunctionDeclContextName(nullptr, &BufferSize);
  if (strlen(FuncNamespaceBegin) != 0) {
    OS << FuncNamespaceBegin;
    OS << "::";
  }
  char *FuncNameBase = Demangler.getFunctionBaseName(nullptr, &BufferSize);
  OS << FuncNameBase;
  return Out;
}

/// Groups the set of annotated values in \p M into instrumentation
/// hooks and intrinsics of instrumentation hooks \n
/// \note This function should get updated as Luthier's programming model
/// gets updated
/// \param [in] M Module to inspect
/// \param [out] Hooks a list of hook functions found in \p M
/// \param [out] Intrinsics a list of intrinsics found in \p M
/// \return any \c llvm::Error encountered during the process
static llvm::Error
getAnnotatedValues(const llvm::Module &M,
                   llvm::SmallVectorImpl<llvm::Function *> &Hooks,
                   llvm::SmallVectorImpl<llvm::Function *> &Intrinsics) {
  const llvm::GlobalVariable *V =
      M.getGlobalVariable("llvm.global.annotations");
  if (V == nullptr)
    return llvm::Error::success();
  const llvm::ConstantArray *CA = cast<llvm::ConstantArray>(V->getOperand(0));
  for (llvm::Value *Op : CA->operands()) {
    auto *CS = cast<llvm::ConstantStruct>(Op);
    // The first field of the struct contains a pointer to the annotated
    // variable.
    llvm::Value *AnnotatedVal = CS->getOperand(0)->stripPointerCasts();
    if (auto *Func = llvm::dyn_cast<llvm::Function>(AnnotatedVal)) {
      // The second field contains a pointer to a global annotation string.
      auto *GV =
          cast<llvm::GlobalVariable>(CS->getOperand(1)->stripPointerCasts());
      llvm::StringRef Content;
      llvm::getConstantStringInfo(GV, Content);
      if (Content == HookAttribute) {
        Hooks.push_back(Func);
        LLVM_DEBUG(llvm::dbgs() << "Found hook " << Func->getName() << ".\n");
      } else if (Content == IntrinsicAttribute) {
        Intrinsics.push_back(Func);
        LLVM_DEBUG(llvm::dbgs()
                   << "Found intrinsic " << Func->getName() << ".\n");
      }
    }
  }
  return llvm::Error::success();
}

llvm::PreservedAnalyses
EmbedInstrumentationModuleBitcodePass::run(llvm::Module &M,
                                           llvm::ModuleAnalysisManager &AM) {
  if (M.getGlobalVariable("llvm.embedded.module", /*AllowInternal=*/true))
    llvm::report_fatal_error(
        "Attempted to embed bitcode twice. Are you passing -fembed-bitcode?",
        /*gen_crash_diag=*/false);

  llvm::Triple T(M.getTargetTriple());
  // Only operate on the AMD GCN code objects
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  // Clone the module in order to preprocess it + not interfere with normal
  // HIP compilation
  auto ClonedModule = llvm::CloneModule(M);

  // Extract all the hooks and intrinsics
  llvm::SmallVector<llvm::Function *, 4> Hooks;
  llvm::SmallVector<llvm::Function *, 4> Intrinsics;
  if (auto Err = getAnnotatedValues(*ClonedModule, Hooks, Intrinsics))
    llvm::report_fatal_error(std::move(Err), true);

  // Remove the annotations variable from the Module now that it is processed
  auto AnnotationGV =
      ClonedModule->getGlobalVariable("llvm.global.annotations");
  if (AnnotationGV) {
    AnnotationGV->dropAllReferences();
    AnnotationGV->eraseFromParent();
  }

  // Remove the llvm.used and llvm.compiler.use variable list
  for (const auto &VarName : {"llvm.compiler.used", "llvm.used"}) {
    auto LLVMUsedVar = ClonedModule->getGlobalVariable(VarName);
    if (LLVMUsedVar != nullptr) {
      LLVMUsedVar->dropAllReferences();
      LLVMUsedVar->eraseFromParent();
    }
  }

  // Give each Hook function a "hook" attribute
  for (auto Hook : Hooks) {
    // TODO: remove the always inline attribute once Hooks support the anyreg
    // calling convention
    Hook->addFnAttr(HookAttribute);
    Hook->removeFnAttr(llvm::Attribute::OptimizeNone);
    Hook->removeFnAttr(llvm::Attribute::NoInline);
    Hook->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  // Remove the body of each intrinsic function and make them extern
  // Also demangle the name and format it similar to LLVM intrinsics
  for (auto Intrinsic : Intrinsics) {
    Intrinsic->deleteBody();
    Intrinsic->setComdat(nullptr);
    llvm::StringRef MangledIntrinsicName = Intrinsic->getName();
    // Format the intrinsic name
    std::string FormattedIntrinsicName;
    llvm::raw_string_ostream FINOS(FormattedIntrinsicName);
    std::string DemangledIntrinsicName =
        getDemangledFunctionNameWithNamespace(MangledIntrinsicName);
    FINOS << DemangledIntrinsicName;
    // Add the output type if it's not void
    auto *ReturnType = Intrinsic->getReturnType();
    if (!ReturnType->isVoidTy()) {
      FINOS << ".";
      ReturnType->print(FINOS);
    }
    // Add the argument types
    for (const auto &Arg : Intrinsic->args()) {
      FINOS << ".";
      Arg.getType()->print(FINOS);
    }
    Intrinsic->addFnAttr(IntrinsicAttribute, DemangledIntrinsicName);
    Intrinsic->setName(FormattedIntrinsicName);
  }

  // Remove all kernels that are meant to serve as a host handle
  for (auto &F : llvm::make_early_inc_range(ClonedModule->functions())) {

    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }

  // Convert all global variables to extern, remove any managed variable
  // initializers
  // Remove any unnecessary variables (e.g. "llvm.metadata")
  // Extract the CUID for identification
  for (auto &GV : llvm::make_early_inc_range(ClonedModule->globals())) {
    auto GVName = GV.getName();
    if (GVName.ends_with(".managed") || GVName == ReservedManagedVar ||
        GV.getSection() == "llvm.metadata") {
      GV.dropAllReferences();
      GV.eraseFromParent();
    } else if (!GVName.starts_with(HipCUIDPrefix)) {
      GV.setInitializer(nullptr);
      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
      GV.setDSOLocal(false);
    }
  }

  auto &LLVMCtx = ClonedModule->getContext();

  auto *Int32Type = llvm::Type::getInt32Ty(LLVMCtx);
  auto *Int32Ptr =
      llvm::PointerType::get(Int32Type, llvm::AMDGPUAS::CONSTANT_ADDRESS);
  // Replace llvm.amdgcn.workgroup.id intrinsics with the luthier-equivalent
  // Remove all kernels that are meant to serve as a host handle
  for (auto &F : llvm::make_early_inc_range(ClonedModule->functions())) {
    for (const auto &[LLVMName, LuthierName, ReturnType] :
         std::initializer_list<
             std::tuple<const char *, const char *, llvm::Type *>>{
             {"llvm.amdgcn.workgroup.id.x", "luthier::workgroupIdX", Int32Type},
             {"llvm.amdgcn.workgroup.idx.y", "luthier::workgroupIdY",
              Int32Type},
             {"llvm.amdgcn.workgroup.idx.z", "luthier::workgroupIdZ",
              Int32Type},
             {"llvm.amdgcn.implicitarg.ptr", "luthier::implicitArgPtr",
              Int32Ptr}}) {
      if (F.getName().starts_with(LLVMName)) {
        for (auto *User : llvm::make_early_inc_range(F.users())) {
          auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User);
          llvm::IRBuilder<> Builder(CallInst);
          auto *LuthierIntrinsicCall = insertCallToIntrinsic(
              *ClonedModule, Builder, LuthierName, *ReturnType);
          CallInst->replaceAllUsesWith(LuthierIntrinsicCall);
          CallInst->eraseFromParent();
        }
        F.dropAllReferences();
        F.eraseFromParent();
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Embedded Module " << ClonedModule->getName()
                          << " dump: ");
  LLVM_DEBUG(ClonedModule->print(llvm::dbgs(), nullptr));

  llvm::SmallVector<char> Data;
  llvm::raw_svector_ostream OS(Data);
  auto PA = llvm::BitcodeWriterPass(OS).run(*ClonedModule, AM);

  llvm::embedBufferInModule(
      M, llvm::MemoryBufferRef(llvm::toStringRef(Data), "ModuleData"),
      ".llvmbc");

  return PA;
}
} // namespace luthier

llvm::PassPluginLibraryInfo getEmbedLuthierBitcodePassPluginInfo() {
  const auto Callback = [](llvm::PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback(
        [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel Opt) {
          MPM.addPass(luthier::EmbedInstrumentationModuleBitcodePass());
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "pre-process-and-embed-luthier-bitcode",
          LLVM_VERSION_STRING, Callback};
}

#ifndef LLVM_LUTHIER_TOOL_COMPILE_PLUGIN_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getEmbedLuthierBitcodePassPluginInfo();
}
#endif
```

The compiler plugin peforms the following actions:
1. It only operates on AMDGPU code objects and not the host executable.
2. It clones the device code's `llvm::Module`. It doesn't interfer with the original compilation  as the next steps will cause Clang/LLVM to be unhappy.
3. Using annotations done in the tool's source code, we identify the hooks and intrinsics inside the device module, and then remove the `llvm::GlobalVariable` that holds the annotated values.
4. Removes the `llvm.compiler.use` and `llvm.used` global variables, as they don't matter in the instrumentation process.
5. Gives each hook device function a "luthier_hook" attribute so that they are easily identified later on. They are also given forced inlined attributes, as hooks are always meant to be inlined at the instrumentation point.
6. Removes the body of all Luthier intrinsic functions, and re-format their CXX Itanium mangled names to look similar to LLVM intrinsics.
7. Removes the "dummy" hook handles defined using the `LUTHIER_EXPORT_HOOK_HANDLE` macro.
8. Makes all the global variables extern, as the non-cloned module will be the one defining them.
9. Replaces a set of LLVM intrinsics with their Luthier equivalents, as they require special treatment during Luthier's instrumentation code generation process.
10. Finally, the cloned module will be embedded inside a non-loadable section of the device code object called `.llvmbc`.






The final result is a stand-alone ELF that can be loaded and run inside the ROCm runtime.

## New Feature: Lowering AMDGPU
On NVIDIA GPUs, accessing this property is very easy; There are dedicated registers the hardware can query to obtain these values; On AMD GPUs, however, there are no dedicated registers for these; Instead, they are either passed as arguments to the SGPRs or

I implemented


## Conclusion and Future Work

Right now I rely solely . However, I need to implement an additional mechanism that compares the kernel arguments in the original kernel and the instrumented one. Even though the