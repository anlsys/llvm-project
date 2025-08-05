# include <xkomp/xkomp.h>

# include "device.h"
# include "omptarget.h"
# include "PluginManager.h"
# include "Shared/APITypes.h"
# include "xktarget.h"

static void
__xktgt_instruction_completed(void * vargs [XKRT_CALLBACK_ARGS_MAX])
{
    xkomp_t * xkomp = (xkomp_t *) vargs[0];
    assert(xkomp);

    task_t * task = (task_t *) vargs[1];
    assert(task);

    xkomp->runtime.task_detachable_decr(task);
}

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

int
__xktgt_target_kernel(
    void *Loc,
    int64_t DeviceId,
    int32_t NumTeams,
    int32_t ThreadLimit,
    void *HostPtr,
    KernelArgsTy *KernelArgs
) {
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
        KernelArgs->NumTeams[0] = NumTeams = 1;

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
    KernelLaunchParamsTy LaunchParams;
    llvm::SmallVector<void *, 16> Args;
    llvm::SmallVector<void *, 16> Ptrs;

    if (KernelArgs->Flags.IsCUDA)
        LOGGER_FATAL("Not supported");
    else
    {
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

        LaunchParams = GenericKernel.prepareArgs(GenericDevice, ArgPtrs, ArgOffsets, KernelArgs->NumArgs, Args, Ptrs, NULL);
    }

    // shared memory for cuda
    const unsigned int sharedmemory = KernelArgs->DynCGroupMem;

    // launch the kernel
    xkrt_device_global_id_t device_global_id = (xkrt_device_global_id_t) (DeviceId + 1);
    xkrt_device_t * device = xkomp->runtime.device_get(device_global_id);
    assert(device);

    xkrt_driver_t * driver = xkomp->runtime.driver_get(device->driver_type);
    assert(driver);

    # if XKOMP_HACK_TARGET_CALL
    // launch kernel
    const xkrt_driver_module_fn_t * fn = (const xkrt_driver_module_fn_t *) GenericKernel.Func;
    driver->f_kernel_launch(
        xkomp_current_stream(),
        xkomp_current_stream_instruction_counter(),
        fn,
        KernelArgs->NumTeams[0],    KernelArgs->NumTeams[1],    KernelArgs->NumTeams[2],
        KernelArgs->ThreadLimit[0], KernelArgs->ThreadLimit[1], KernelArgs->ThreadLimit[2],
        sharedmemory,
        LaunchParams.Data,  // array of pointer
        LaunchParams.Size   // size of array in bytes
    );
    # else /* XKOMP_HACK_TARGET_CALL */
    LOGGER_FATAL("TODO");
    # endif /* XKOMP_HACK_TARGET_CALL */

    LOGGER_FATAL("TODO: Launch kernel");

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
    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    # if XKOMP_HACK_TARGET_CALL
    task_t * task = xkomp_current_task();
    assert(task);
    # else /* XKOMP_HACK_TARGET_CALL */
    LOGGER_FATAL("`XKOMP_HACK_TARGET_CALL` disabled - enable it or implement codegen in llvm to pass the stream and instruction index to target calls");
    # endif /* XKOMP_HACK_TARGET_CALL */

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

        // launch instruction
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

        // if map(to: _)
        if (ArgType & OMP_TGT_MAPTYPE_TO)
        {
            // increment counter, as we are submitting an instruction, to defer
            // task completion to instruction completion
            xkomp->runtime.task_detachable_incr(task);

            // retrieve xkrt device
            const xkrt_device_global_id_t dst_device_global_id = (xkrt_device_global_id_t) (DeviceId + 1);
            xkrt_device_t * device = xkomp->runtime.device_get(dst_device_global_id);
            assert(device);

            const xkrt_device_global_id_t src_device_global_id = HOST_DEVICE_GLOBAL_ID;

            xkrt_callback_t callback;
            callback.func = __xktgt_instruction_completed;
            callback.args[0] = xkomp;
            callback.args[1] = task;

            device->offloader_stream_instruction_submit_copy<size_t, uintptr_t>(
                (size_t) ArgSize,
                dst_device_global_id,
                (const uintptr_t) TgtPtrBegin,
                src_device_global_id,
                (const uintptr_t) HstPtrBegin,
                callback
            );
        }

        // if map(from: _)
        if (ArgType & OMP_TGT_MAPTYPE_FROM)
        {
            LOGGER_FATAL("TODO: submit a k-instruction here - memcpy1d(dst=HstPtrBegin, src=TgtPtrBegin, size=ArgSize");
        }
    }
}
