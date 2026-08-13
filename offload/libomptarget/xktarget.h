#ifndef __XKTARGET_H__
# define __XKTARGET_H__

# include "Shared/APITypes.h"

int __xktgt_target_kernel(
    void *Loc,
    int64_t DeviceId,
    int32_t NumTeams,
    int32_t ThreadLimit,
    void *HostPtr,
    KernelArgsTy *KernelArgs
);

int __xktgt_target_kernel_nowait(
    void *Loc,
    int64_t DeviceId,
    int32_t NumTeams,
    int32_t ThreadLimit,
    void *HostPtr,
    KernelArgsTy *KernelArgs
);

void __xktgt_target_data_update_nowait_mapper(
    void *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList
);

void __xktgt_target_data_update_mapper(
    void *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames,
    void **ArgMappers
);

/// Redirect device memory (de)allocation to the XKRT allocator. Called from
/// DeviceTy::allocData / DeviceTy::deleteData. Return true if XKRT handled the
/// request (a device-kind allocation on an initialized XKRT device); false if
/// the caller must fall back to the plugin allocator (host/unified kinds,
/// non-XKRT devices, or allocations issued before the XKRT device is ready).
bool __xktgt_data_alloc(int32_t DeviceId, int64_t Size, int32_t Kind, void **OutPtr);
bool __xktgt_data_delete(int32_t DeviceId, void *Ptr, int32_t Kind);

#endif /* __XKTARGET_H__ */
