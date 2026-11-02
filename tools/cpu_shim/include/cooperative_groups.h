#pragma once
#include "pfshim_core.h"
namespace cooperative_groups {
struct coalesced_group {
    unsigned long long thread_rank() const { return 0; }
    unsigned long long size() const { return 1; }
    template <class T> T shfl(T v, int) const { return v; }
};
inline coalesced_group coalesced_threads() { return {}; }
}
