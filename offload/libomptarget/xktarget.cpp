# include <xkomp/xkomp.h>

# include "device.h"
# include "omptarget.h"
# include "PluginManager.h"
# include "Shared/APITypes.h"
# include "xktarget.h"

XKRT_NAMESPACE_USE;

TableMap *getTableMap(void *HostPtr);

////////////////
// omp target //
////////////////

KernelArgsTy * upgradeKernelArgs(
    KernelArgsTy *KernelArgs,
    KernelArgsTy &LocalKernelArgs,
    int32_t NumTeams,
    int32_t ThreadLimit
);

extern "C" int omp_get_default_device(void);
extern "C" int omp_get_default_device(void);
extern "C" xkrt_device_unique_id_t omp_device_id_to_xkomp(int device_id);

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

int
__xktgt_target_kernel(
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
        KernelArgs->NumTeams[0] = NumTeams = 1;
    }

    // 'KernelArgs' will point to 'LocalKernelArgs' if it becomes upgraded, else it remains unchanged
    KernelArgsTy LocalKernelArgs;
    KernelArgs = upgradeKernelArgs(KernelArgs, LocalKernelArgs, NumTeams, ThreadLimit);

    // Get device/plugin
    auto DeviceOrErr = PM->getDevice(DeviceId);
    if (!DeviceOrErr)
        LOGGER_FATAL("Invalid device");
    DeviceTy & Device = *DeviceOrErr;

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
    for (int32_t i = 0; i < NumClangLaunchArgs ; ++i)
    {
        assert(KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TARGET_PARAM);
        void *HstPtrBegin = KernelArgs->ArgPtrs[i];
        void *HstPtrBase = KernelArgs->ArgBasePtrs[i];
        void *TgtPtrBegin;
        ptrdiff_t TgtBaseOffset;
        TargetPointerResultTy TPR;

        if (KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL)
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
    KernelLaunchParamsTy LaunchParams = GenericKernel.prepareArgs(GenericDevice, ArgPtrs, ArgOffsets, KernelArgs->NumArgs, Args, Ptrs, KernelLaunchEnvironment);

    // shared memory for cuda
    // const unsigned int sharedmemory = KernelArgs->DynCGroupMem;

    uint32_t NumThreads[3] = {KernelArgs->ThreadLimit[0], KernelArgs->ThreadLimit[1], KernelArgs->ThreadLimit[2]};
    uint32_t NumBlocks[3] = {KernelArgs->NumTeams[0], KernelArgs->NumTeams[1], KernelArgs->NumTeams[2]};
    if (!GenericKernel.isBareMode())
    {
        NumThreads[0] = GenericKernel.getNumThreads(GenericDevice, NumThreads);
        NumBlocks[0]  = GenericKernel.getNumBlocks(GenericDevice, NumBlocks, KernelArgs->Tripcount, NumThreads[0], KernelArgs->ThreadLimit[0] > 0);
    }

    // launch the kernel
    const device_unique_id_t device_unique_id = omp_device_id_to_xkomp(DeviceId);

    // device_t * device = xkomp->runtime.device_get(device_unique_id);
    // assert(device);

    // driver_t * driver = xkomp->runtime.driver_get(device->driver_type);
    // assert(driver);

    // TODO: support shared memory

    constexpr command_queue_type_t qtype = XKRT_QUEUE_TYPE_KERN;
    constexpr ocg::command_type_t  ctype = ocg::COMMAND_TYPE_PROG;
    constexpr command_flag_t       flags = COMMAND_FLAG_NONE;
    xkomp->runtime.task_emit_command(
        device_unique_id,
        qtype,
        ctype,
        flags,
        [&] (command_t * command) {
            command->prog.launcher.variadic.fn        = GenericKernel.Func;
            command->prog.launcher.variadic.args      = LaunchParams.Data;
            command->prog.launcher.variadic.args_size = LaunchParams.Size;
            command->prog.grid.x                      = NumBlocks[0];
            command->prog.grid.y                      = NumBlocks[1];
            command->prog.grid.z                      = NumBlocks[2];
            command->prog.block.x                     = NumThreads[0];
            command->prog.block.y                     = NumThreads[1];
            command->prog.block.z                     = NumThreads[2];
        }
    );

    return 0;
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
        assert(!(ArgTypes[i] & OMP_TGT_MAPTYPE_NON_CONTIG));

        // launch command
        void * HstPtrBegin = Args[i];
        int64_t ArgSize = ArgSizes[i];
        int64_t ArgType = ArgTypes[i];

        TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(HstPtrBegin, ArgSize, /*UpdateRefCount=*/false, /*UseHoldRefCount=*/false, /*MustContain=*/true);
        void * TgtPtrBegin = TPR.TargetPointer;

        if (!TPR.isPresent())
            LOGGER_FATAL("Data is not mapped");

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
            const ocg::command_type_t  ctype = (ArgType & OMP_TGT_MAPTYPE_TO) ? ocg::COMMAND_TYPE_COPY_H2D_1D : ocg::COMMAND_TYPE_COPY_D2H_1D;
            constexpr command_flag_t   flags = COMMAND_FLAG_NONE;

            xkomp->runtime.task_emit_command(
                device_unique_id,
                qtype,
                ctype,
                flags,
                [&] (command_t * command) {
                    command->copy_1D.src_device_unique_id   = src_device_unique_id;
                    command->copy_1D.dst_device_unique_id   = dst_device_unique_id;
                    command->copy_1D.src_device_addr        = src_ptr;
                    command->copy_1D.dst_device_addr        = dst_ptr;
                    command->copy_1D.size                   = (size_t) ArgSize;
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

        if (!TPR.isPresent())
            LOGGER_FATAL("Data is not mapped");

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
            const ocg::command_type_t ctype = (ArgType & OMP_TGT_MAPTYPE_TO) ? ocg::COMMAND_TYPE_COPY_H2D_1D : ocg::COMMAND_TYPE_COPY_D2H_1D;
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
