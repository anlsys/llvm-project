# include <xkomp/xkomp.h>

# include "device.h"
# include "omptarget.h"
# include "PluginManager.h"
# include "xktarget.h"

static void
__xktgt_instruction_completed(const void * vargs [XKRT_CALLBACK_ARGS_MAX])
{
    xkomp_t * xkomp = (xkomp_t *) vargs[0];
    assert(xkomp);

    task_t * task = (task_t *) vargs[1];
    assert(task);

    xkomp->runtime.task_detachable_decr(task);
}

EXTERN
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
    void * xktask
) {
    xkomp_t * xkomp = xkomp_get();
    assert(xkomp);

    task_t * task = (task_t *) xktask;
    assert(task);

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
