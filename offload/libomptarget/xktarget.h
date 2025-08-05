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

void __xktgt_target_data_update_nowait_mapper(
    void *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList
);

#endif /* __XKTARGET_H__ */
