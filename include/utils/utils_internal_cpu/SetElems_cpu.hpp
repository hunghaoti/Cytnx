#ifndef _H_SetElems_cpu_
#define _H_SetElems_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include <vector>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx{
    namespace utils_internal{


        void SetElems_cpu_cdtcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdtcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdtd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdtf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdtu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cdtu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_cftcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cftcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cftd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cftf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cfti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cftu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cfti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_cftu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_dtcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dtcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dtd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dtf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dtu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_dtu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_ftcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_ftcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_ftd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_ftf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_fti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_ftu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_fti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_ftu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_i64tcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64tcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64td(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64tf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64ti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64tu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64ti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i64tu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_u64tcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64tcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64td(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64tf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64ti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64tu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64ti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u64tu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_i32tcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32tcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32td(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32tf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32ti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32tu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32ti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_i32tu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);

        void SetElems_cpu_u32tcd(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32tcf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32td(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32tf(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32ti64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32tu64(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32ti32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);
        void SetElems_cpu_u32tu32(void *in, void *out,const std::vector<cytnx_uint64> &offj, const std::vector<cytnx_uint64> &new_offj, const std::vector<std::vector<cytnx_uint64> >&locators, const cytnx_uint64 &TotalElem,const bool & is_scalar);


    }
}
#endif
