#ifndef _H_Bond_
#define _H_Bond_

#include "Type.hpp"
#include "Symmetry.hpp"
#include "cytnx_error.hpp"
#include <initializer_list>
#include <vector>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"
namespace cytnx{

    enum bondType: int{
        BD_KET = -1,
        BD_BRA = 1,
        BD_REG =0
    };

    class Bond_impl: public intrusive_ptr_base<Bond_impl>{
        private:
            cytnx_uint64 _dim;
            bondType _type;
            std::vector< std::vector<cytnx_int64> > _qnums;
            std::vector<Symmetry> _syms;

        public:

            Bond_impl(): _type(bondType::BD_REG) {};   
            //Bond_impl(const cytnx_uint64 &dim, const std::initializer_list<std::initializer_list<cytnx_int64> > &in_qnums, const std::initializer_list<Symmetry> &in_syms={}, const bondType &bd_type=bondType::BD_REG);
            //Bond_impl(const cytnx_uint64 &dim, const std::vector<std::vector<cytnx_int64> > &in_qnums, const std::vector<Symmetry> &in_syms={}, const bondType &bd_type=bondType::BD_REG);

            void Init(const cytnx_uint64 &dim, const std::vector<std::vector<cytnx_int64> > &in_qnums = {{}}, const std::vector<Symmetry> &in_syms={}, const bondType &bd_type=bondType::BD_REG);

            void Init(const cytnx_uint64 &dim, const std::initializer_list<std::initializer_list<cytnx_int64> > &in_qnums = {{}}, const std::initializer_list<Symmetry> &in_syms={}, const bondType &bd_type=bondType::BD_REG);



            bondType                                type() const& {return this->_type;};
            std::vector<std::vector<cytnx_int64> > qnums() const& {return this->_qnums;};
            cytnx_uint64                             dim() const &{return this->_dim;};
            cytnx_uint32                            Nsym() const &{return this->_syms.size();};
            std::vector<Symmetry>                   syms() const &{return vec_clone(this->_syms);};


            void set_type(const bondType &new_bondType){
                this->_type = new_bondType;
            }

            void clear_type(){
                this->_type = bondType::BD_REG;
            }

            
            boost::intrusive_ptr<Bond_impl> clone(){
                boost::intrusive_ptr<Bond_impl> out(new Bond_impl());
                out->_dim = this->dim();
                out->_type = this->type();
                out->_qnums = this->qnums();
                out->_syms  = this->syms();// return a clone of vec!
                return out;
            }

        

    };//Bond_impl


    //wrapper:
    class Bond{
        public:
            boost::intrusive_ptr<Bond_impl> _impl;
            Bond(): _impl(new Bond_impl()){};
            Bond(const cytnx_uint64 &dim, const std::initializer_list<std::initializer_list<cytnx_int64> > &in_qnums={}, const std::initializer_list<Symmetry> &in_syms={}, const bondType &bd_type=bondType::BD_REG): _impl(new Bond_impl()){
                this->_impl->Init(dim,in_qnums,in_syms,bd_type);
            };
            Bond(const cytnx_uint64 &dim, const std::vector<std::vector<cytnx_int64> > &in_qnums={}, const std::vector<Symmetry> &in_syms={}, const bondType &bd_type=bondType::BD_REG): _impl(new Bond_impl()){
                this->_impl->Init(dim,in_qnums,in_syms,bd_type);
            };

            bondType                                type() const& {return this->_impl->type();};
            std::vector<std::vector<cytnx_int64> > qnums() const& {return this->_impl->qnums();};
            cytnx_uint64                             dim() const &{return this->_impl->dim();};
            cytnx_uint32                            Nsym() const &{return this->_impl->syms().size();};
            std::vector<Symmetry>                   syms() const &{return this->_impl->syms();};


            void set_type(const bondType &new_bondType){
                this->_impl->set_type(new_bondType);
            }

            void clear_type(){
                this->_impl->clear_type();
            }

            Bond clone() const{
                Bond out;
                out._impl = this->_impl->clone();
                return out;
            }

    };


    std::ostream& operator<<(std::ostream &os,const Bond &bin);

}



#endif
