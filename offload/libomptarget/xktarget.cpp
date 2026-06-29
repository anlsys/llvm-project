# include <xkomp/xkomp.h>

# include "device.h"
# include "omptarget.h"
# include "PluginManager.h"
# include "Shared/APITypes.h"
# include "xktarget.h"

# include "llvm/Support/Error.h"

# include <mutex>
# include <string>
# include <unordered_map>

XKRT_NAMESPACE_USE;

/// External function provided by the xkomp runtime.
/// Given the index of an access clause expression on a target construct,
/// returns the corresponding device pointer. The index is 0-based and
/// counts across all access clause expressions in order of appearance.
extern "C" void *xkomp_access_pointer(int idx);

TableMap *getTableMap(void *HostPtr);

////////////////
// omp target //
////////////////

struct UpgradedArgBuffersTy {
  llvm::SmallVector<void *, 0> BasePtrs;
  llvm::SmallVector<void *, 0> Ptrs;
  llvm::SmallVector<int64_t, 0> Sizes;
  llvm::SmallVector<int64_t, 0> Types;
  llvm::SmallVector<map_var_info_t, 0> Names;
  llvm::SmallVector<void *, 0> Mappers;
};

KernelArgsTy * upgradeKernelArgs(
    KernelArgsTy * KernelArgs,
    KernelArgsTy & LocalKernelArgs,
    UpgradedArgBuffersTy & Bufs,
    int32_t NumTeams,
    int32_t ThreadLimit
);

extern "C" int omp_get_default_device(void);
extern "C" int omp_get_default_device(void);
extern "C" xkrt_device_unique_id_t omp_device_id_to_xkomp(int device_id);

static void
__xktgt_target_kernel_launch_free_dup_args(void * args[XKRT_CALLBACK_ARGS_MAX])
{
    free(args[0]);
}

/// Read the device-side LLVM-IR that clang embedded for a target kernel as the
/// device global "<kernel>__ir" (see emitTargetKernelSourceIR). It is read
/// host-side from the device image (no device memory access, no JIT), cached per
/// kernel for the process lifetime. Sets {OutRaw, OutSize} to {NULL, 0} when no
/// such global exists (e.g. a TU compiled without IR forwarding).
static void
__xktgt_get_kernel_ir(
    llvm::omp::target::plugin::GenericPluginTy & Plugin,
    llvm::omp::target::plugin::GenericDeviceTy & GenericDevice,
    llvm::omp::target::plugin::GenericKernelTy & Kernel,
    void *& OutRaw,
    size_t & OutSize
) {
    using namespace llvm::omp::target::plugin;

    /* per-kernel cache (the buffer is owned here, for the process lifetime) */
    static std::mutex Mtx;
    static std::unordered_map<const void *, std::pair<void *, size_t>> Cache;

    const void * Key = (const void *) &Kernel;
    {
        std::lock_guard<std::mutex> Guard(Mtx);
        auto It = Cache.find(Key);
        if (It != Cache.end())
        {
            OutRaw  = It->second.first;
            OutSize = It->second.second;
            return ;
        }
    }

    void * Raw  = NULL;
    size_t Size = 0;

    GenericGlobalHandlerTy & GH    = Plugin.getGlobalHandler();
    DeviceImageTy          & Image = Kernel.getImage();
    std::string              Name  = std::string(Kernel.getName()) + "__ir";

    /* resolve size/presence from the image ELF (host-side) */
    GlobalTy Meta(Name);
    if (llvm::Error E = GH.getGlobalMetadataFromImage(GenericDevice, Image, Meta))
    {
        llvm::consumeError(std::move(E));   /* no IR embedded for this kernel */
    }
    else if (Meta.getSize() > 0)
    {
        void * Buf = malloc(Meta.getSize());
        if (Buf)
        {
            GlobalTy Host(Name, Meta.getSize(), Buf);
            if (llvm::Error E2 = GH.readGlobalFromImage(GenericDevice, Image, Host))
            {
                llvm::consumeError(std::move(E2));
                free(Buf);
            }
            else
            {
                Raw  = Buf;
                Size = Meta.getSize();
            }
        }
    }

    {
        std::lock_guard<std::mutex> Guard(Mtx);
        // another thread may have populated the entry while we read unlocked;
        // if so, drop our buffer and use the existing one (avoids a leak)
        auto It = Cache.find(Key);
        if (It != Cache.end())
        {
            if (Raw)
                free(Raw);
            OutRaw  = It->second.first;
            OutSize = It->second.second;
            return ;
        }
        Cache[Key] = { Raw, Size };
    }
    OutRaw  = Raw;
    OutSize = Size;
}

/// Implements a kernel entry that executes the target region on the specified
/// device.
///
/// \param Loc Source location associated with this target region.
/// \param DeviceId The device to execute this region, -1 indicated the default.
/// \param NumTeams Number of teams to launch the region with, -1 indicates a
///                 non-teams region and 0 indicates it was unspecified.
/// \param ThreadLimit Limit to the number of threads to use in the kernel
///                    launch, 0 indicates it was unspecified.
/// \param HostPtr  The pointer to the host function registered with the kernel.
/// \param Args     All arguments to this kernel launch (see struct definition).

template <bool nowait>
static int
__xktgt_target_kernel_launch(
    void *Loc,
    int64_t DeviceId,
    int32_t NumTeams,
    int32_t ThreadLimit,
    void *HostPtr,
    KernelArgsTy *KernelArgs
) {
    assert(KernelArgs);

    if (DeviceId == -1)
        DeviceId = omp_get_default_device();

    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    // TODO: map to
    // TODO: firstprivate

    // Get device/plugin — must happen before getTableMap() because
    // PM->getDevice() triggers loadImagesOntoDevice() which populates
    // the TargetsTable entries needed below.
    auto DeviceOrErr = PM->getDevice(DeviceId);
    if (!DeviceOrErr)
        LOGGER_FATAL("Invalid device");
    DeviceTy & Device = *DeviceOrErr;

    // get device function pointer
    TableMap *TM = getTableMap(HostPtr);
    __tgt_target_table *TargetTable = nullptr;
    {
        std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
        assert(TM->Table->TargetsTable.size() > (size_t)DeviceId);
        TargetTable = TM->Table->TargetsTable[DeviceId];
    }
    assert(TargetTable);

    void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].Address;
    assert(TgtEntryPtr);

    bool IsTeams = NumTeams != -1;
    if (!IsTeams)
    {
        KernelArgs->UserNumBlocks[0] = NumTeams = 1;
    }

    // 'KernelArgs' will point to 'LocalKernelArgs' if it becomes upgraded, else it remains unchanged
    KernelArgsTy LocalKernelArgs;
    UpgradedArgBuffersTy UpgradedBufs;
    KernelArgs = upgradeKernelArgs(KernelArgs, LocalKernelArgs, UpgradedBufs, NumTeams, ThreadLimit);

    GenericPluginTy * GenericPlugin = Device.RTL;
    assert(GenericPlugin);

    using GenericDeviceTy = llvm::omp::target::plugin::GenericDeviceTy;
    GenericDeviceTy & GenericDevice = GenericPlugin->getDevice(DeviceId);

    using GenericKernelTy = llvm::omp::target::plugin::GenericKernelTy;
    GenericKernelTy & GenericKernel = *reinterpret_cast<GenericKernelTy *>(TgtEntryPtr);

    // pack args to pass to the kernel launch
    llvm::SmallVector<void *, 16> Args;
    llvm::SmallVector<void *, 16> Ptrs;

    if (KernelArgs->Flags.IsCUDA)
        LOGGER_FATAL("Not supported");

    // processDataBefore
    llvm::SmallVector<void *> TgtArgs;
    llvm::SmallVector<ptrdiff_t> TgtOffsets;

    int NumClangLaunchArgs = KernelArgs->NumArgs;
    int AccessIdx = 0; // running index for access clause expressions
    for (int32_t i = 0; i < NumClangLaunchArgs ; ++i)
    {
        assert(KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TARGET_PARAM);
        void *HstPtrBegin = KernelArgs->ArgPtrs[i];
        void *HstPtrBase = KernelArgs->ArgBasePtrs[i];
        void *TgtPtrBegin;
        ptrdiff_t TgtBaseOffset;
        TargetPointerResultTy TPR;

        if (KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_ACCESS)
        {
            // Access clause entry: resolve via xkomp_access_pointer(idx)
            // instead of the standard host-to-device mapping lookup.
            // Compute offset between base pointer and section start so the
            // kernel receives the correct base pointer (e.g., for v[-1:n+2],
            // xkomp returns &dev_v[-1], offset adjusts it back to dev_v).
            TgtPtrBegin = xkomp_access_pointer(AccessIdx++);
            TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
        }
        else if (KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL)
        {
            TgtPtrBegin = HstPtrBase;
            TgtBaseOffset = 0;
        }
        else if (KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_PRIVATE)
        {
            TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
            const bool IsFirstPrivate = (KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO);
            if (IsFirstPrivate)
                LOGGER_FATAL("Not supported");
            TgtPtrBegin = NULL;
            TgtBaseOffset = 0;
        }
        else
        {
            if (KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
                HstPtrBase = *reinterpret_cast<void **>(HstPtrBase);
            TPR = DeviceOrErr->getMappingInfo().getTgtPtrBegin(
                    HstPtrBegin, KernelArgs->ArgSizes[i],
                    /*UpdateRefCount=*/false,
                    /*UseHoldRefCount=*/false);
            TgtPtrBegin = TPR.TargetPointer;
            TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
        }
        TgtArgs.push_back(TgtPtrBegin);
        TgtOffsets.push_back(TgtBaseOffset);
    }

    void ** ArgPtrs = TgtArgs.data();
    ptrdiff_t * ArgOffsets = TgtOffsets.data();

    KernelArgs->NumArgs = TgtArgs.size();

    // getKernelLaunchEnvironment
    // assert(!GenericKernel.KernelEnvironment.Configuration.ReductionDataSize ||
    //         !GenericKernel.KernelEnvironment.Configuration.ReductionBufferLength);

    KernelLaunchEnvironmentTy * KernelLaunchEnvironment = reinterpret_cast<KernelLaunchEnvironmentTy *>(~0);
    assert(KernelLaunchEnvironment);

    // prepareArgs
    KernelLaunchParamsTy LaunchParams = GenericKernel.prepareArgs(GenericDevice, ArgPtrs, ArgOffsets, KernelArgs->NumArgs, Args, Ptrs, KernelLaunchEnvironment, KernelArgs->Version);

    // shared memory for cuda
    // const unsigned int sharedmemory = KernelArgs->DynCGroupMem;

    uint32_t NumThreads[3] = {KernelArgs->UserThreadLimit[0], KernelArgs->UserThreadLimit[1], KernelArgs->UserThreadLimit[2]};
    uint32_t NumBlocks[3] = {KernelArgs->UserNumBlocks[0], KernelArgs->UserNumBlocks[1], KernelArgs->UserNumBlocks[2]};
    if (!GenericKernel.isBareMode())
    {
        NumThreads[0] = GenericKernel.getEffectiveNumThreads(GenericDevice, NumThreads[0]);
        NumBlocks[0]  = GenericKernel.getEffectiveNumBlocks(GenericDevice, NumBlocks[0], KernelArgs->Tripcount, NumThreads[0], KernelArgs->UserThreadLimit[0] > 0);
    }

    // HIP (unlike CUDA) rejects a kernel launch when any grid or block
    // dimension is 0, returning hipErrorInvalidValue. Unspecified y/z
    // dimensions arrive here as 0 (the front-end fills only index [0] for
    // 1-D launches). Normalize every dimension to at least 1 so the launch is
    // valid on all backends.
    for (int d = 0; d < 3; ++d)
    {
        if (NumThreads[d] == 0) NumThreads[d] = 1;
        if (NumBlocks[d]  == 0) NumBlocks[d]  = 1;
    }

    // launch the kernel
    const device_unique_id_t device_unique_id = omp_device_id_to_xkomp(DeviceId);

    // device_t * device = xkomp->runtime.device_get(device_unique_id);
    // assert(device);

    // driver_t * driver = xkomp->runtime.driver_get(device->driver_type);
    // assert(driver);

    // TODO: support shared memory

    constexpr command_queue_type_t qtype = XKRT_QUEUE_TYPE_KERN;
    constexpr cgir::command_type_t  ctype = cgir::COMMAND_TYPE_PROG;
    constexpr command_flag_t       flags = COMMAND_FLAG_NONE;

    // device kernel LLVM-IR embedded by clang (read host-side from the image,
    // cached). {NULL,0} if the TU was compiled without IR forwarding. The buffer
    // is owned by the cache (process lifetime), so the command is non-owning.
    void * KernelIR     = NULL;
    size_t KernelIRSize = 0;
    __xktgt_get_kernel_ir(*GenericPlugin, GenericDevice, GenericKernel, KernelIR, KernelIRSize);

    const auto builder = [&] (command_t * cmd) {
        cmd->prog.launcher.variadic.fn        = GenericKernel.Func;
        cmd->prog.launcher.variadic.args      = LaunchParams.Data;
        cmd->prog.launcher.variadic.args_size = LaunchParams.Size;
        cmd->prog.source.type                 = cgir::COMMAND_PROG_SOURCE_TYPE_LLVMIR;
        cmd->prog.source.content.llvmir.raw    = KernelIR;
        cmd->prog.source.content.llvmir.size   = KernelIRSize;
        cmd->prog.source.content.llvmir._owned = false;   // owned by the cache
        cmd->prog.source.content.llvmir.symbol = KernelIR ? GenericKernel.getName() : NULL;
        cmd->prog.grid.x                      = NumBlocks[0];
        cmd->prog.grid.y                      = NumBlocks[1];
        cmd->prog.grid.z                      = NumBlocks[2];
        cmd->prog.block.x                     = NumThreads[0];
        cmd->prog.block.y                     = NumThreads[1];
        cmd->prog.block.z                     = NumThreads[2];
    };

    // if no wait, emit a command (e.g., with an event and increasing detach counter)
    if (nowait)
    {
        // gotta dupplicate heap-allocated args, and free on command completion
        const auto builder_nowait = [&] (command_t * command)
        {
            // construct command
            builder(command);

            // TODO: can be do faster than a malloc here?
            // idea: have a `small_vector_t` within the `command_t`

            // dupplicate args
            void * dup_args = malloc(LaunchParams.Size);
            assert(dup_args);
            memcpy(dup_args, LaunchParams.Data, LaunchParams.Size);

            command->prog.launcher.variadic.args = dup_args;

            // TODO: reenable this free, but gotta find a way to handle replay in TDG
            # if 0
            // set callback to release args
            callback_t cb;
            cb.func = __xktgt_target_kernel_launch_free_dup_args;
            cb.args[0] = dup_args;
            command->completion_callback_push(cb);
            # endif
        };

        xkomp->runtime.task_emit_command(device_unique_id, qtype, ctype, flags, builder_nowait);
    }
    // else, submit serialized command
    else
    {
        // can use heap-allocated args
        constexpr command_flag_t flags = COMMAND_FLAG_SERIALIZED | COMMAND_FLAG_SYNCHRONOUS;
        command_t command(ctype, flags);
        builder(&command);
        command.prog.launcher.variadic.args = LaunchParams.Data;
        xkomp->runtime.command_submit(device_unique_id, &command);
    }

    return 0;
}

int
__xktgt_target_kernel_nowait(
    void *Loc,
    int64_t DeviceId,
    int32_t NumTeams,
    int32_t ThreadLimit,
    void *HostPtr,
    KernelArgsTy *KernelArgs
) {
    return __xktgt_target_kernel_launch<true>(Loc, DeviceId, NumTeams, ThreadLimit, HostPtr, KernelArgs);
}

int
__xktgt_target_kernel(
    void *Loc,
    int64_t DeviceId,
    int32_t NumTeams,
    int32_t ThreadLimit,
    void *HostPtr,
    KernelArgsTy *KernelArgs
) {
    return __xktgt_target_kernel_launch<false>(Loc, DeviceId, NumTeams, ThreadLimit, HostPtr, KernelArgs);
}

//////////////////////////////
// omp target update nowait //
//////////////////////////////

void
__xktgt_target_data_update_nowait_mapper(
    void *Loc,
    int64_t DeviceId,
    int32_t ArgNum,
    void ** ArgsBase,
    void ** Args,
    int64_t * ArgSizes,
    int64_t * ArgTypes,
    void ** ArgNames,
    void ** ArgMappers,
    int32_t DepNum,
    void * DepList,
    int32_t NoAliasDepNum,
    void * NoAliasDepList
) {
    if (DeviceId == -1)
        DeviceId = omp_get_default_device();

    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    auto DeviceOrErr = PM->getDevice(DeviceId);
    if (!DeviceOrErr)
        LOGGER_FATAL("Could not get device %ld - %s", DeviceId, toString(DeviceOrErr.takeError()).c_str());
    DeviceTy & Device = *DeviceOrErr;

    for (int i = 0 ; i < ArgNum ; ++i)
    {
        if ((ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL) || (ArgTypes[i] & OMP_TGT_MAPTYPE_PRIVATE))
            continue ;

        // mapper
        if (ArgMappers && ArgMappers[i])
            LOGGER_FATAL("Custom mapper not supported");

        // only support continuous transfer for now
        if (ArgTypes[i] & OMP_TGT_MAPTYPE_NON_CONTIG)
            LOGGER_FATAL("Non-contiguous transfer not supported");

        // launch command
        void * HstPtrBegin = Args[i];
        int64_t ArgSize = ArgSizes[i];
        int64_t ArgType = ArgTypes[i];

        # if 0
        void * HstPtrBase  = ArgsBase[i];
        int64_t offset = ((int64_t) HstPtrBegin - (int64_t) HstPtrBase);
        TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(HstPtrBase, ArgSize, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false, /*MustContain=*/true);
        void * TgtPtrBegin = (void *) ((uintptr_t)TPR.TargetPointer + offset);
        # else
        TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(HstPtrBegin, ArgSize, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false, /*MustContain=*/true);
        void * TgtPtrBegin = TPR.TargetPointer;
        # endif

        if (!TPR.isPresent()) {
            // Match vanilla LLVM behavior (omptarget.cpp targetDataContiguous):
            // unmapped data in target update is a noop unless 'present' modifier.
            if (ArgType & OMP_TGT_MAPTYPE_PRESENT)
                LOGGER_FATAL("device mapping required by 'present' motion modifier "
                             "does not exist for host address %p (%" PRId64 " bytes)",
                             HstPtrBegin, ArgSize);
            LOGGER_DEBUG("hst data %p not found in mapping, becomes a noop", HstPtrBegin);
            continue ;
        }

        if (TPR.Flags.IsHostPointer)
        {
            LOGGER_DEBUG("Unified memory - transfer is a no-op");
            return ;
        }

        // if map(to: _) or map(from: _)
        if ((ArgType & OMP_TGT_MAPTYPE_TO) || (ArgType & OMP_TGT_MAPTYPE_FROM))
        {
            // retrieve xkrt device
            const device_unique_id_t device_unique_id = omp_device_id_to_xkomp(DeviceId);

            // src/dst devices
            const device_unique_id_t src_device_unique_id = (ArgType & OMP_TGT_MAPTYPE_TO) ? XKRT_HOST_DEVICE_UNIQUE_ID : device_unique_id;
            const device_unique_id_t dst_device_unique_id = (ArgType & OMP_TGT_MAPTYPE_TO) ? device_unique_id           : XKRT_HOST_DEVICE_UNIQUE_ID;

            // src/dst pointers
            const uintptr_t dst_ptr = (const uintptr_t) ((ArgType & OMP_TGT_MAPTYPE_TO) ? TgtPtrBegin : HstPtrBegin);
            const uintptr_t src_ptr = (const uintptr_t) ((ArgType & OMP_TGT_MAPTYPE_TO) ? HstPtrBegin : TgtPtrBegin);

            // queue/command type
            const command_queue_type_t qtype = (ArgType & OMP_TGT_MAPTYPE_TO) ? XKRT_QUEUE_TYPE_H2D      : XKRT_QUEUE_TYPE_D2H;
            const cgir::command_type_t  ctype = (ArgType & OMP_TGT_MAPTYPE_TO) ? cgir::COMMAND_TYPE_COPY_H2D_1D : cgir::COMMAND_TYPE_COPY_D2H_1D;
            constexpr command_flag_t   flags = COMMAND_FLAG_NONE;

            xkomp->runtime.task_emit_command(
                device_unique_id,
                qtype,
                ctype,
                flags,
                [&] (command_t * cmd) {
                    cmd->copy_1D.src_device_unique_id   = src_device_unique_id;
                    cmd->copy_1D.dst_device_unique_id   = dst_device_unique_id;
                    cmd->copy_1D.src_device_addr        = src_ptr;
                    cmd->copy_1D.dst_device_addr        = dst_ptr;
                    cmd->copy_1D.size                   = (size_t) ArgSize;
                }
            );
        }
    }
}

///////////////////////
// omp target update //
///////////////////////

void
__xktgt_target_data_update_mapper(
    void *Loc,
    int64_t DeviceId,
    int32_t ArgNum,
    void ** ArgsBase,
    void ** Args,
    int64_t * ArgSizes,
    int64_t * ArgTypes,
    void ** ArgNames,
    void ** ArgMappers
) {
    if (DeviceId == -1)
        DeviceId = omp_get_default_device();

    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    auto DeviceOrErr = PM->getDevice(DeviceId);
    if (!DeviceOrErr)
        LOGGER_FATAL("Could not get device %ld - %s", DeviceId, toString(DeviceOrErr.takeError()).c_str());
    DeviceTy & Device = *DeviceOrErr;

    for (int i = 0 ; i < ArgNum ; ++i)
    {
        if ((ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL) || (ArgTypes[i] & OMP_TGT_MAPTYPE_PRIVATE))
            continue ;

        // mapper
        if (ArgMappers && ArgMappers[i])
            LOGGER_FATAL("Custom mapper not supported");

        // only support continuous transfer for now
        assert(!(ArgTypes[i] & OMP_TGT_MAPTYPE_NON_CONTIG));

        // launch command
        void * HstPtrBegin = Args[i];
        int64_t ArgSize = ArgSizes[i];
        int64_t ArgType = ArgTypes[i];

        TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(HstPtrBegin, ArgSize, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false, /*MustContain=*/true);
        void * TgtPtrBegin = TPR.TargetPointer;

        if (!TPR.isPresent()) {
            if (ArgType & OMP_TGT_MAPTYPE_PRESENT)
                LOGGER_FATAL("device mapping required by 'present' motion modifier "
                             "does not exist for host address %p (%" PRId64 " bytes)",
                             HstPtrBegin, ArgSize);
            LOGGER_DEBUG("hst data %p not found in mapping, becomes a noop", HstPtrBegin);
            continue ;
        }

        if (TPR.Flags.IsHostPointer)
        {
            LOGGER_DEBUG("Unified memory - transfer is a no-op");
            return ;
        }

        // if map(to: _) or map(from: _)
        if ((ArgType & OMP_TGT_MAPTYPE_TO) || (ArgType & OMP_TGT_MAPTYPE_FROM))
        {
            // retrieve xkrt device
            const device_unique_id_t device_unique_id = omp_device_id_to_xkomp(DeviceId);
            assert(device_unique_id != XKRT_HOST_DEVICE_UNIQUE_ID);

            // src/dst devices
            const device_unique_id_t src_device_unique_id = (ArgType & OMP_TGT_MAPTYPE_TO) ? XKRT_HOST_DEVICE_UNIQUE_ID : device_unique_id;
            const device_unique_id_t dst_device_unique_id = (ArgType & OMP_TGT_MAPTYPE_TO) ? device_unique_id           : XKRT_HOST_DEVICE_UNIQUE_ID;

            // src/dst pointers
            const uintptr_t dst_ptr = (const uintptr_t) ((ArgType & OMP_TGT_MAPTYPE_TO) ? TgtPtrBegin : HstPtrBegin);
            const uintptr_t src_ptr = (const uintptr_t) ((ArgType & OMP_TGT_MAPTYPE_TO) ? HstPtrBegin : TgtPtrBegin);

            // queue/command type
            const cgir::command_type_t ctype = (ArgType & OMP_TGT_MAPTYPE_TO) ? cgir::COMMAND_TYPE_COPY_H2D_1D : cgir::COMMAND_TYPE_COPY_D2H_1D;
            constexpr command_flag_t flags = COMMAND_FLAG_SERIALIZED | COMMAND_FLAG_SYNCHRONOUS;

            // create and submit serialized command
            command_t command(ctype, flags);
            command.copy_1D.src_device_unique_id = src_device_unique_id;
            command.copy_1D.dst_device_unique_id = dst_device_unique_id;
            command.copy_1D.src_device_addr      = src_ptr;
            command.copy_1D.dst_device_addr      = dst_ptr;
            command.copy_1D.size                 = (size_t) ArgSize;
            xkomp->runtime.command_submit(device_unique_id, &command);
        }
    }
}
