#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include <iostream>
#include <vector>
#include <string>
using namespace std;

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_UvT) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Svd] error, Svd can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Svd] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_uint64 n_singlu = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      // std::cout << n_singlu << std::endl;

      Tensor U, S, vT;
      S.Init({n_singlu}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
             in.device());  // if type is complex, S should be real
      // S.storage().set_zeros();
      if (is_UvT) {
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
        // U.storage().set_zeros();
      }
      if (is_UvT) {
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
        // vT.storage().set_zeros();
      }

      if (Tin.device() == Device.cpu) {
        // cytnx::linalg_internal::lii.Svd_ii[in.dtype()](
        //   in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
        //   S._impl->storage()._impl, in.shape()[0], in.shape()[1]);
        cytnx::linalg_internal::lii.Sdd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_UvT) {
          out.push_back(U);
          out.push_back(vT);
        }

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuSvd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_UvT) {
          // cout << "Original:\n" << in << endl;
          // cout << "S:\n" << S << endl;
          // cout << "Recompose1!:\n" << Matmul(Matmul(U, Diag(S)), vT) << endl;
          // cout << "Recompose2!:\n"
          //      << Tensordot(Tensordot(U, Diag(S), {1}, {0}), vT, {1}, {0}) << endl;
          out.push_back(U);
          out.push_back(vT);
        }

        return out;
  #else
        cytnx_error_msg(true, "[Svd] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
  #endif
      }
    }

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {

    // actual impls:
    void _svd_Dense_UT(std::vector<cytnx::UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                       const bool &compute_uv) {
      //[Note] outCyT must be empty!

      // DenseUniTensor:
      // cout << "entry Dense UT" << endl;

      Tensor tmp;
      if (Tin.is_contiguous())
        tmp = Tin.get_block_();
      else {
        tmp = Tin.get_block();
        tmp.contiguous_();
      }

      vector<cytnx_uint64> tmps = tmp.shape();
      vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      vector<string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp.reshape_({rowdim, -1});

      vector<Tensor> outT = cytnx::linalg::Svd(tmp, compute_uv);
      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[t].shape()[0]);

      Cy_S.Init({newBond, newBond}, {std::string("_aux_L"), std::string("_aux_R")}, 1, Type.Double,
                Device.cpu, true);  // it is just reference so no hurt to alias ^^

      // cout << "[AFTER INIT]" << endl;
      Cy_S.put_block_(outT[t]);
      t++;

      if (compute_uv) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        vector<cytnx_int64> shapeU = vec_clone(oldshape, Tin.rowrank());
        shapeU.push_back(-1);
        outT[t].reshape_(shapeU);
        Cy_U.Init(outT[t], false, Tin.rowrank());
        vector<string> labelU = vec_clone(oldlabel, Tin.rowrank());
        labelU.push_back(Cy_S.labels()[0]);
        Cy_U.set_labels(labelU);
        t++;  // U
      }
      if (compute_uv) {
        cytnx::UniTensor &Cy_vT = outCyT[t];
        vector<cytnx_int64> shapevT(Tin.rank() - Tin.rowrank() + 1);
        shapevT[0] = -1;
        memcpy(&shapevT[1], &oldshape[Tin.rowrank()], sizeof(cytnx_int64) * (shapevT.size() - 1));

        outT[t].reshape_(shapevT);
        Cy_vT.Init(outT[t], false, 1);
        // cout << shapevT.size() << endl;
        vector<string> labelvT(shapevT.size());
        labelvT[0] = Cy_S.labels()[1];
        // memcpy(&labelvT[1], &oldlabel[Tin.rowrank()], sizeof(cytnx_int64) * (labelvT.size() -
        // 1));
        std::copy(oldlabel.begin() + Tin.rowrank(), oldlabel.end(), labelvT.begin() + 1);
        Cy_vT.set_labels(labelvT);
        t++;  // vT
      }
      // if tag, then update  the tagging informations
      if (Tin.is_tag()) {
        Cy_S.tag();
        t = 1;
        if (compute_uv) {
          cytnx::UniTensor &Cy_U = outCyT[t];
          Cy_U._impl->_is_tag = true;
          for (int i = 0; i < Cy_U.rowrank(); i++) {
            Cy_U.bonds()[i].set_type(Tin.bonds()[i].type());
          }
          Cy_U.bonds().back().set_type(cytnx::BD_BRA);
          Cy_U._impl->_is_braket_form = Cy_U._impl->_update_braket();
          t++;
        }
        if (compute_uv) {
          cytnx::UniTensor &Cy_vT = outCyT[t];
          Cy_vT._impl->_is_tag = true;
          Cy_vT.bonds()[0].set_type(cytnx::BD_KET);
          for (int i = 1; i < Cy_vT.rank(); i++) {
            Cy_vT.bonds()[i].set_type(Tin.bonds()[Tin.rowrank() + i - 1].type());
          }
          Cy_vT._impl->_is_braket_form = Cy_vT._impl->_update_braket();
          t++;
        }

      }  // if tag
    }

    void _svd_Block_UT(std::vector<cytnx::UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                       const bool &compute_uv) {
	  std::cout << "start block" << std::endl;

    }  // _svd_Block_UT

    std::vector<cytnx::UniTensor> Svd(const cytnx::UniTensor &Tin, const bool &is_UvT) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[Svd][ERROR] Svd for UniTensor should have rank>1 and rowrank>0%s", "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[Svd][ERROR] Svd for diagonal UniTensor is trivial and currently not "
                      "support. Use other manipulation.%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _svd_Dense_UT(outCyT, Tin, is_UvT);

      } else if (Tin.uten_type() == UTenType.Block) {
        _svd_Block_UT(outCyT, Tin, is_UvT);

      } else {
        cytnx_error_msg(true, "[ERROR] only support svd for Dense and Block UniTensor.%s", "\n");

      }  // is block form ?

      return outCyT;

    };  // Svd

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
