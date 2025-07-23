#include "tenseal/cpp/tensors/ckkstensor.h"


#include <stdio.h>
#include <math.h>



#include <iostream>
#include <ctime> 

namespace tenseal {

using namespace seal;
using namespace std;

CKKSTensor::CKKSTensor(const shared_ptr<TenSEALContext>& ctx,
                       const PlainTensor<double>& tensor,
                       std::optional<double> scale, bool batch) {
    printf("## Tenseal libreria fichero ckkstensor.cpp\n");
    this->link_tenseal_context(ctx);
    if (scale.has_value()) {
        this->_init_scale = scale.value();
    } else {
        this->_init_scale = ctx->global_scale();
    }


    vector<Ciphertext> enc_data;
    vector<size_t> enc_shape = tensor.shape();
    auto data = tensor.batch(0);
    size_t size;
    if (batch) {
        _batch_size = enc_shape[0];
        enc_shape.erase(enc_shape.begin());
        size = tensor.batch(0).size();
    } else {
        size = tensor.flat_size();
    }
    enc_data.resize(size);
    task_t worker_func = [&](size_t start, size_t end) -> bool {
        if (batch) {
            for (size_t i = start; i < end; i++) {
                enc_data[i] =
                    CKKSTensor::encrypt(ctx, this->_init_scale, data.at(i));
            }
        } else {
            for (size_t i = start; i < end; i++) {
                enc_data[i] = CKKSTensor::encrypt(ctx, this->_init_scale,
                                                  tensor.flat_at(i));
            }
        }

        return true;
    };

    this->dispatch_jobs(worker_func, size);

    _data = TensorStorage<Ciphertext>(enc_data, enc_shape);
}

CKKSTensor::CKKSTensor(const shared_ptr<TenSEALContext>& ctx,
                       const string& tensor) {
    this->link_tenseal_context(ctx);
    this->load(tensor);
}

CKKSTensor::CKKSTensor(const string& tensor) { this->load(tensor); }

CKKSTensor::CKKSTensor(const TenSEALContextProto& ctx,
                       const CKKSTensorProto& tensor) {
    this->load_context_proto(ctx);
    this->load_proto(tensor);
}

CKKSTensor::CKKSTensor(const shared_ptr<TenSEALContext>& ctx,
                       const CKKSTensorProto& tensor) {
    this->link_tenseal_context(ctx);
    this->load_proto(tensor);
}

CKKSTensor::CKKSTensor(const shared_ptr<const CKKSTensor>& tensor) {
    this->link_tenseal_context(tensor->tenseal_context());
    this->_init_scale = tensor->scale();
    this->_data = TensorStorage<Ciphertext>(tensor->data(), tensor->shape());
    this->_batch_size = tensor->_batch_size;
}

CKKSTensor::CKKSTensor(const shared_ptr<const CKKSTensor>& tensor,
                       const TensorStorage<Ciphertext>& storage) {
    this->link_tenseal_context(tensor->tenseal_context());
    this->_init_scale = tensor->scale();
    this->_data = storage;
    this->_batch_size = tensor->_batch_size;
    printf("El batch size de tenseal es: %ld\n", this->_batch_size );
}


/*------------------------------------------------------------------------------------------ CKKSTensor */

CKKSTensor::CKKSTensor(const shared_ptr<const CKKSTensor>& tensor,
                       const vector<Ciphertext>& storage, const vector<size_t>& shape) {
    this->clear();
    this->link_tenseal_context(tensor->tenseal_context());
    this->_init_scale = tensor->scale();
    this->_data = TensorStorage<Ciphertext>(storage, shape);
    this->_batch_size = tensor->_batch_size;
    printf("Constructor 2\n");
    
}

CKKSTensor::CKKSTensor(const shared_ptr<const CKKSTensor>& tensor,
                       const  xt::xarray<Ciphertext>& storage) {
    this->link_tenseal_context(tensor->tenseal_context());
    this->_init_scale = tensor->scale();
    this->_data = TensorStorage<Ciphertext>(storage);
    this->_batch_size = tensor->_batch_size;
    printf("Constructor 3\n");
}  


CKKSTensor::CKKSTensor(const shared_ptr<const CKKSTensor>& tensor,
                       const vector<vector<Ciphertext>>& storage, const vector<size_t>& shape) {
    printf("4\n" );
    this->link_tenseal_context(tensor->tenseal_context());
    this->_init_scale = tensor->scale();
    this->_data = TensorStorage<Ciphertext>(storage, shape[0], shape[1]);
    this->_batch_size = tensor->_batch_size;
}
/*---------------------------------------------------------------------------------------*/
CKKSTensor::CKKSTensor(const shared_ptr<TenSEALContext>& ctx,
                       const vector<vector<Ciphertext>>& storage, const int size,  const vector<size_t>& shape) {
    this->link_tenseal_context(ctx);
    this->_init_scale = ctx->global_scale();
    this->_data = TensorStorage<Ciphertext>(storage, shape, shape[0]);
    printf("el no batch de tenseal es %ld\n", shape[0]);
    this->_batch_size = size;
    printf("el batch de tenseal es %d\n", size);
}

CKKSTensor::CKKSTensor(const shared_ptr<TenSEALContext>& ctx,
                       const vector<vector<Ciphertext>>& storage, const int size) {
    this->link_tenseal_context(ctx);
    this->_init_scale = ctx->global_scale();
    this->_data = TensorStorage<Ciphertext>(storage);
    //printf("el batch de tenseal es %d\n", size);
    this->_batch_size = size;
}

CKKSTensor::CKKSTensor(const shared_ptr<const CKKSTensor>& tensor,  const int height, const int width, const int padding) {   
     this->_data = TensorStorage<Ciphertext>(height, width, padding);
}

   


/*------------------------------------------------------------------------------------------*/

Ciphertext CKKSTensor::encrypt(const shared_ptr<TenSEALContext>& ctx,
                               const double scale, const vector<double>& data) {
    if (data.empty()) {
        throw invalid_argument("Attempting to encrypt an empty vector");
    }
    auto slot_count = ctx->slot_count<CKKSEncoder>();
    if (data.size() > slot_count)
        // number of slots available is poly_modulus_degree / 2
        throw invalid_argument(
            "can't encrypt vectors of this size, please use a larger "
            "polynomial modulus degree.");

    Ciphertext ciphertext(*ctx->seal_context());
    Plaintext plaintext;
    ctx->encode<CKKSEncoder>(data, plaintext, scale);
    ctx->encrypt(plaintext, ciphertext);

    return ciphertext;
}

Ciphertext CKKSTensor::encrypt(const shared_ptr<TenSEALContext>& ctx,
                               const double scale, const double data) {
    Ciphertext ciphertext(*ctx->seal_context());
    Plaintext plaintext;
    ctx->encode<CKKSEncoder>(data, plaintext, scale);
    ctx->encrypt(plaintext, ciphertext);

    return ciphertext;
}

PlainTensor<double> CKKSTensor::decrypt(const shared_ptr<SecretKey>& sk) const {
    Plaintext plaintext;
    auto sz = this->_data.flat_size();
    auto shape = this->shape_with_batch();

    if (_batch_size) {
        vector<vector<double>> result;
        result.reserve(sz);

        for (auto it = _data.cbegin(); it != _data.cend(); it++) {
            vector<double> buff;
            this->tenseal_context()->decrypt(*sk, *it, plaintext);
            this->tenseal_context()->decode<CKKSEncoder>(plaintext, buff);
            result.push_back(
                vector<double>(buff.begin(), buff.begin() + *_batch_size));
        }

        return PlainTensor<double>(/*batched_tensor=*/result,
                                   /*shape_with_batch=*/shape,
                                   /*batch_axis=*/0);
    } else {
        vector<double> result;
        result.reserve(sz);

        for (auto it = _data.cbegin(); it != _data.cend(); it++) {
            vector<double> buff;
            this->tenseal_context()->decrypt(*sk, *it, plaintext);
            this->tenseal_context()->decode<CKKSEncoder>(plaintext, buff);
            result.push_back(buff[0]);
        }

        return PlainTensor<double>(result, /*shape_with_batch=*/shape);
    }
}

shared_ptr<CKKSTensor> CKKSTensor::negate_inplace() {
    for (auto& ct : _data)
        this->tenseal_context()->evaluator->negate_inplace(ct);
    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::square_inplace() {
   
    for (auto& ct : _data) {
        this->tenseal_context()->evaluator->square_inplace(ct);
        this->auto_relin(ct);
        this->auto_rescale(ct);
    }
    
    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::power_inplace(unsigned int power) {
    if (power == 0) {
        auto ones = PlainTensor<double>::repeat_value(1, this->shape());
        *this = CKKSTensor(this->tenseal_context(), ones, this->_init_scale,
                           _batch_size.has_value());
        return shared_from_this();
    }

    if (power == 1) {
        return shared_from_this();
    }

    if (power == 2) {
        this->square_inplace();
        return shared_from_this();
    }

    int closest_power_of_2 = 1 << static_cast<int>(floor(log2(power)));
    power -= closest_power_of_2;
    if (power == 0) {
        this->power_inplace(closest_power_of_2 / 2)->square_inplace();
    } else {
        auto closest_pow2_vector = this->power(closest_power_of_2);
        this->power_inplace(power)->mul_inplace(closest_pow2_vector);
    }

    return shared_from_this();
}

void CKKSTensor::perform_op(seal::Ciphertext& ct, seal::Ciphertext other,
                            OP op) {
    this->auto_same_mod(other, ct);
    switch (op) {
        case OP::ADD:
            this->tenseal_context()->evaluator->add_inplace(ct, other);
            break;
        case OP::SUB:
            this->tenseal_context()->evaluator->sub_inplace(ct, other);
            break;
        case OP::MUL:
            this->tenseal_context()->evaluator->multiply_inplace(ct, other);
            this->auto_relin(ct);
            this->auto_rescale(ct);
            break;
        default:
            throw invalid_argument("operation not defined");
    }
}

void CKKSTensor::perform_plain_op(seal::Ciphertext& ct, seal::Plaintext other,
                                  OP op) {
    this->auto_same_mod(other, ct);
    switch (op) {
        case OP::ADD:
            this->tenseal_context()->evaluator->add_plain_inplace(ct, other);
            break;
        case OP::SUB:
            this->tenseal_context()->evaluator->sub_plain_inplace(ct, other);
            break;
        case OP::MUL:
            try {
                this->tenseal_context()->evaluator->multiply_plain_inplace(
                    ct, other);
            } catch (const std::logic_error& e) {
                if (strcmp(e.what(), "result ciphertext is transparent") == 0) {
                    // replace by encryption of zero
                    this->tenseal_context()->encrypt_zero(ct);
                    ct.scale() = this->_init_scale;
                } else {  // Something else, need to be forwarded
                    throw;
                }
            }
           
            this->auto_relin(ct);
            this->auto_rescale(ct);
         
            break;
        default:
            throw invalid_argument("operation not defined");
    }
}

shared_ptr<CKKSTensor> CKKSTensor::op_inplace(
    const shared_ptr<CKKSTensor>& raw_operand, OP op) {
    auto operand = raw_operand;
    if (this->shape() != operand->shape()) {
        operand = this->broadcast_or_throw(operand);
    }

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        for (size_t i = start; i < end; i++) {
            this->perform_op(this->_data.flat_ref_at(i),
                             operand->_data.flat_ref_at(i), op);
        }
        return true;
    };

    this->dispatch_jobs(worker_func, this->_data.flat_size());

    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::op_plain_inplace(
    const PlainTensor<double>& raw_operand, OP op) {
    // TODO batched ops

    auto operand = raw_operand;
    if (this->shape() != operand.shape()) {
        operand = this->broadcast_or_throw<>(operand);
    }

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        Plaintext plaintext;
        for (size_t i = start; i < end; i++) {
            this->tenseal_context()->encode<CKKSEncoder>(
                operand.flat_at(i), plaintext, this->_init_scale);
            this->perform_plain_op(this->_data.flat_ref_at(i), plaintext, op);
        }
        return true;
    };

    this->dispatch_jobs(worker_func, this->_data.flat_size());

    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::op_plain_inplace(const double& operand,
                                                    OP op) {
    Plaintext plaintext;
    this->tenseal_context()->encode<CKKSEncoder>(operand, plaintext,
                                                 this->_init_scale);

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        for (size_t i = start; i < end; i++) {
            this->perform_plain_op(this->_data.flat_ref_at(i), plaintext, op);
        }
        return true;
    };

    this->dispatch_jobs(worker_func, this->_data.flat_size());

    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::add_inplace(
    const shared_ptr<CKKSTensor>& to_add) {
    return this->op_inplace(to_add, OP::ADD);
}

shared_ptr<CKKSTensor> CKKSTensor::sub_inplace(
    const shared_ptr<CKKSTensor>& to_sub) {
    return this->op_inplace(to_sub, OP::SUB);
}

shared_ptr<CKKSTensor> CKKSTensor::mul_inplace(
    const shared_ptr<CKKSTensor>& to_mul) {
    return this->op_inplace(to_mul, OP::MUL);
}

shared_ptr<CKKSTensor> CKKSTensor::add_plain_inplace(
    const PlainTensor<double>& to_add) {
    return this->op_plain_inplace(to_add, OP::ADD);
}

shared_ptr<CKKSTensor> CKKSTensor::sub_plain_inplace(
    const PlainTensor<double>& to_sub) {
    return this->op_plain_inplace(to_sub, OP::SUB);
}

shared_ptr<CKKSTensor> CKKSTensor::mul_plain_inplace(
    const PlainTensor<double>& to_mul) {
    return this->op_plain_inplace(to_mul, OP::MUL);
}

shared_ptr<CKKSTensor> CKKSTensor::add_plain_inplace(const double& to_add) {
    return this->op_plain_inplace(to_add, OP::ADD);
}

shared_ptr<CKKSTensor> CKKSTensor::sub_plain_inplace(const double& to_sub) {
    return this->op_plain_inplace(to_sub, OP::SUB);
}

shared_ptr<CKKSTensor> CKKSTensor::mul_plain_inplace(const double& to_mul) {
    return this->op_plain_inplace(to_mul, OP::MUL);
}

shared_ptr<CKKSTensor> CKKSTensor::sum_inplace(size_t axis) {
    if (axis >= shape_with_batch().size())
        throw invalid_argument("invalid axis");

    if (_batch_size && axis == 0) return sum_batch_inplace();

    if (_batch_size) axis--;

    auto new_shape = shape();
    auto new_len = _data.flat_size() / shape()[axis];

    // remove the summation axis
    new_shape.erase(new_shape.begin() + axis);
    auto working_shape = shape();

    std::vector<Ciphertext> new_data(new_len);
    vector<vector<Ciphertext>> batches(new_len);

    auto old_strides = _data.strides();
    xt::xarray<double> dummy(new_shape);
    auto new_strides = dummy.strides();

    for (size_t idx = 0; idx < _data.flat_size(); ++idx) {
        auto pos = position_from_strides(old_strides, idx);
        pos.erase(pos.begin() + axis);

        size_t new_idx = 0;
        for (size_t pidx = 0; pidx < pos.size(); ++pidx)
            new_idx += new_strides[pidx] * pos[pidx];

        batches[new_idx].push_back(_data.flat_ref_at(idx));
    }

    for (size_t idx = 0; idx < new_len; ++idx) {
        tenseal_context()->evaluator->add_many(batches[idx], new_data[idx]);
    }

    _data = TensorStorage<Ciphertext>(new_data, new_shape);
    return shared_from_this();
}
shared_ptr<CKKSTensor> CKKSTensor::sum_batch_inplace() {
    if (!_batch_size) throw invalid_argument("unsupported operation");

    for (size_t idx = 0; idx < _data.flat_size(); ++idx) {
        sum_vector(this->tenseal_context(), _data.flat_ref_at(idx),
                   *_batch_size);
    }

    _batch_size = {};
    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::polyval_inplace(
    const vector<double>& coefficients) {
    if (coefficients.size() == 0) {
        throw invalid_argument(
            "the coefficients vector need to have at least one element");
    }

    int degree = static_cast<int>(coefficients.size()) - 1;
    while (degree >= 0) {
        if (coefficients[degree] == 0.0)
            degree--;
        else
            break;
    }

    if (degree == -1) {
        auto zeros =
            PlainTensor<double>::repeat_value(0, this->shape_with_batch());
        *this = CKKSTensor(this->tenseal_context(), zeros, this->_init_scale,
                           _batch_size.has_value());
        return shared_from_this();
    }

    // pre-compute squares of x
    auto x = this->copy();

    int max_square = static_cast<int>(floor(log2(degree)));
    vector<shared_ptr<CKKSTensor>> x_squares;
    x_squares.reserve(max_square + 1);
    x_squares.push_back(x->copy());  // x
    for (int i = 1; i <= max_square; i++) {
        x->square_inplace();
        x_squares.push_back(x->copy());  // x^(2^i)
    }

    auto cst_coeff = PlainTensor<double>::repeat_value(
        coefficients[0], this->shape_with_batch());
    auto result =
        CKKSTensor::Create(this->tenseal_context(), cst_coeff,
                           this->_init_scale, _batch_size.has_value());

    // coefficients[1] * x + ... + coefficients[degree] * x^(degree)
    for (int i = 1; i <= degree; i++) {
        if (coefficients[i] == 0.0) continue;
        x = compute_polynomial_term(i, coefficients[i], x_squares);
        result->add_inplace(x);
    }

    this->_data = TensorStorage<Ciphertext>(result->data(), result->shape());
    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::dot_inplace(
    const shared_ptr<CKKSTensor>& other) {
    auto this_shape = this->shape();
    auto other_shape = other->shape();
    if (this_shape.size() == 1) {
        if (other_shape.size() == 1) {  // 1D-1D
            // inner product
            this->_mul_inplace(other);
            this->sum_inplace();
            return shared_from_this();
        } else if (other_shape.size() == 2) {  // 1D-2D
            if (this_shape[0] != other_shape[0])
                throw invalid_argument("can't perform dot: dimension mismatch");
            this->reshape_inplace(vector<size_t>({this_shape[0], 1}));
            this->_mul_inplace(other);
            this->sum_inplace();
            return shared_from_this();
        } else {
            throw invalid_argument(
                "don't support dot operations of more than 2 dimensions");
        }
    } else if (this_shape.size() == 2) {
        if (other_shape.size() == 1) {  // 2D-1D
            if (this_shape[1] != other_shape[0])
                throw invalid_argument("can't perform dot: dimension mismatch");
            auto other_copy =
                other->reshape(vector<size_t>({1, other_shape[0]}));
            this->_mul_inplace(other_copy);
            this->sum_inplace(1);
            return shared_from_this();
        } else if (other_shape.size() == 2) {  // 2D-2D
            this->_matmul_inplace(other);
            return shared_from_this();
        } else {
            throw invalid_argument(
                "don't support dot operations of more than 2 dimensions");
        }
    } else {
        throw invalid_argument(
            "don't support dot operations of more than 2 dimensions");
    }
}

shared_ptr<CKKSTensor> CKKSTensor::dot_plain_inplace(
    const PlainTensor<double>& other) {
    auto this_shape = this->shape();
    auto other_shape = other.shape();
    if (this_shape.size() == 1) {
        if (other_shape.size() == 1) {  // 1D-1D
            // inner product
            this->_mul_inplace(other);
            this->sum_inplace();
            return shared_from_this();
        } else if (other_shape.size() == 2) {  // 1D-2D
            if (this_shape[0] != other_shape[0])
                throw invalid_argument("can't perform dot: dimension mismatch");
            this->reshape_inplace(vector<size_t>({this_shape[0], 1}));
            this->_mul_inplace(other);
            this->sum_inplace();
            return shared_from_this();
        } else {
            throw invalid_argument(
                "don't support dot operations of more than 2 dimensions");
        }
    } else if (this_shape.size() == 2) {
        if (other_shape.size() == 1) {  // 2D-1D
            if (this_shape[1] != other_shape[0])
                throw invalid_argument("can't perform dot: dimension mismatch");
            auto other_copy = other;
            other_copy.reshape_inplace(vector<size_t>({1, other_shape[0]}));
            this->_mul_inplace(other_copy);
            this->sum_inplace(1);
            return shared_from_this();
        } else if (other_shape.size() == 2) {  // 2D-2D
            this->_matmul_inplace(other);
            return shared_from_this();
        } else {
            throw invalid_argument(
                "don't support dot operations of more than 2 dimensions");
        }
    } else {
        throw invalid_argument(
            "don't support dot operations of more than 2 dimensions");
    }
}

shared_ptr<CKKSTensor> CKKSTensor::matmul_inplace(
    const shared_ptr<CKKSTensor>& other) {
    auto this_shape = this->shape();
    auto other_shape = other->shape();

    if (this_shape.size() != 2)
        throw invalid_argument("this tensor isn't a matrix");
    if (other_shape.size() != 2)
        throw invalid_argument("operand tensor isn't a matrix");
    if (this_shape[1] != other_shape[0])
        throw invalid_argument("can't multiply matrices");  // put matrix shapes

    vector<size_t> new_shape = vector({this_shape[0], other_shape[1]});
    size_t new_size = new_shape[0] * new_shape[1];
    vector<Ciphertext> new_data;
    new_data.resize(new_shape[0] * new_shape[1]);

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        vector<Ciphertext> to_sum;
        to_sum.resize(this_shape[1]);
        for (size_t i = start; i < end; i++) {
            auto evaluator = this->tenseal_context()->evaluator;
            size_t row = i / new_shape[1];
            size_t col = i % new_shape[1];
            // inner product
            for (size_t j = 0; j < this_shape[1]; j++) {
                to_sum[j] = this->_data.at({row, j});
                this->perform_op(to_sum[j], other->_data.at({j, col}), OP::MUL);
            }
            Ciphertext acc(*this->tenseal_context()->seal_context(),
                           to_sum[0].parms_id());
            evaluator->add_many(to_sum, acc);
            // set element[row, col] to the computed inner product
            new_data[i] = acc;
        }
        return true;
    };

    this->dispatch_jobs(worker_func, new_size);

    this->_data = TensorStorage(new_data, new_shape);
    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::matmul_plain_inplace( // original
    const PlainTensor<double>& other) {
    auto this_shape = this->shape();
    auto other_shape = other.shape();
    
    printf("## matmul_plain_inplace ckkkstensor.cpp %f\n", this->_init_scale);

    if (this_shape.size() != 2)
        throw invalid_argument("this tensor isn't a matrix");
    if (other_shape.size() != 2)
        throw invalid_argument("operand tensor isn't a matrix");
    if (this_shape[1] != other_shape[0])
        throw invalid_argument("can't multiply matrices");  // put matrix shapes

    vector<size_t> new_shape = vector({this_shape[0], other_shape[1]});
    size_t new_size = new_shape[0] * new_shape[1];
    vector<Ciphertext> new_data;
    new_data.resize(new_shape[0] * new_shape[1]);

    printf("## new shape[0] %ld and new shape[1] %ld\n",  new_shape[0] , new_shape[1]);

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        vector<Ciphertext> to_sum;
        to_sum.resize(this_shape[1]);
        for (size_t i = start; i < end; i++) {
            auto evaluator = this->tenseal_context()->evaluator;
            size_t row = i / new_shape[1];
            size_t col = i % new_shape[1];
            // inner product
            for (size_t j = 0; j < this_shape[1]; j++) {
                to_sum[j] = this->_data.at({row, j});
                Plaintext pt;
                this->tenseal_context()->encode<CKKSEncoder>(
                    other.at({j, col}), pt, this->_init_scale);
                this->perform_plain_op(to_sum[j], pt, OP::MUL);
            }
            Ciphertext acc(*this->tenseal_context()->seal_context(),
                           to_sum[0].parms_id());
            evaluator->add_many(to_sum, acc);
            // set element[row, col] to the computed inner product
            new_data[i] = acc;
        }
        return true;
    };

    this->dispatch_jobs(worker_func, new_size);
    this->_data = TensorStorage<Ciphertext>();
    this->_data = TensorStorage(new_data, new_shape);
    return shared_from_this();
}

/*--------------------------------------------------------------------------*/

shared_ptr<CKKSTensor> CKKSTensor::matmul_plain_inplace_row_(
    const PlainTensor<double>& other) {
    auto this_shape = this->shape();
    auto other_shape = other.shape();
    

    if (this_shape.size() != 2)
        throw invalid_argument("this tensor isn't a matrix");
    if (other_shape.size() != 2)
        throw invalid_argument("operand tensor isn't a matrix");
    if (this_shape[1] != other_shape[1])
        throw invalid_argument("can't multiply matrices");  // put matrix shapes

    vector<size_t> new_shape = vector({this_shape[0], other_shape[0]});
    size_t new_size = new_shape[0] * new_shape[1];
    vector<Ciphertext> new_data;
    new_data.resize(new_shape[0] * new_shape[1]);

    printf("## new shape[0] %ld and new shape[1] %ld\n",  new_shape[0] , new_shape[1]);

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        vector<Ciphertext> to_sum;
        to_sum.resize(this_shape[1]);
        for (size_t i = start; i < end; i++) {
            auto evaluator = this->tenseal_context()->evaluator;
            size_t row_1 = i / new_shape[1];
            size_t row_2 = i % new_shape[1];
            // inner product
            for (size_t j = 0; j < this_shape[1]; j++) {
                to_sum[j] = this->_data.at({row_1, j});
                Plaintext pt;
                this->tenseal_context()->encode<CKKSEncoder>(
                    other.at({row_2, j}), pt, this->_init_scale);
                this->perform_plain_op(to_sum[j], pt, OP::MUL);
            }
            Ciphertext acc(*this->tenseal_context()->seal_context(),
                           to_sum[0].parms_id());
            evaluator->add_many(to_sum, acc);
            // set element[row, col] to the computed inner product
            new_data[i] = acc;
        }
        return true;
    };

    this->dispatch_jobs(worker_func, new_size);
    this->_data = TensorStorage<Ciphertext>();
    this->_data = TensorStorage(new_data, new_shape);
    return shared_from_this();
}


/*--------------------------------------------------------------------------*/

CKKSTensor CKKSTensor::matmul_plain_inplace_row(
    const PlainTensor<double>& other) {
    auto this_shape = this->shape();
    auto other_shape = other.shape();
    

    if (this_shape.size() != 2)
        throw invalid_argument("this tensor isn't a matrix");
    if (other_shape.size() != 2)
        throw invalid_argument("operand tensor isn't a matrix");
    if (this_shape[1] != other_shape[1])
        throw invalid_argument("can't multiply matrices");  // put matrix shapes

    vector<size_t> new_shape = vector({this_shape[0], other_shape[0]});
    size_t new_size = new_shape[0] * new_shape[1];
    vector<Ciphertext> new_data;
    new_data.resize(new_shape[0] * new_shape[1]);

    printf("## new shape[0] %ld and new shape[1] %ld\n",  new_shape[0] , new_shape[1]);

    task_t worker_func = [&](size_t start, size_t end) -> bool {
        vector<Ciphertext> to_sum;
        to_sum.resize(this_shape[1]);
        for (size_t i = start; i < end; i++) {
            auto evaluator = this->tenseal_context()->evaluator;
            size_t row_1 = i / new_shape[1];
            size_t row_2 = i % new_shape[1];
            // inner product
            for (size_t j = 0; j < this_shape[1]; j++) {
                to_sum[j] = this->_data.at({row_1, j});
                Plaintext pt;
                this->tenseal_context()->encode<CKKSEncoder>(
                    other.at({row_2, j}), pt, this->_init_scale);
                this->perform_plain_op(to_sum[j], pt, OP::MUL);
            }
            Ciphertext acc(*this->tenseal_context()->seal_context(),
                           to_sum[0].parms_id());
            evaluator->add_many(to_sum, acc);
            // set element[row, col] to the computed inner product
            new_data[i] = acc;
        }
        return true;
    };

    this->dispatch_jobs(worker_func, new_size);
    clock_t start= clock();
    CKKSTensor newTensor = CKKSTensor(shared_from_this(), new_data, new_shape);
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f\n", time_taken);
    return newTensor;
}



/*-------------------------------------------------------------------------*/


CKKSTensor CKKSTensor::subscript(const vector<pair<size_t, size_t>>& pairs) {
    TensorStorage<Ciphertext> storage = this->_data.subscript(pairs);
    CKKSTensor newTensor = CKKSTensor(shared_from_this(), storage);
    return newTensor;
}





/*------------------------------------------------------------------------*/ //IMROW


CKKSTensor CKKSTensor::im2row(const int kernel, const int stride, const int channels, const int padding, const int output_channels) {
    auto this_shape = this->shape(); 
    auto original_height = this_shape[0];
    // -----------------------------------------
    int height=sqrt(original_height);
    printf("height: %ld, original_height: %ld\n", height, original_height);
    int width = height;
    // -----------------------------------------

    int height_w_padding = height  + 2*padding;
    int total_rows =pow(((height_w_padding - kernel + stride)/stride), 2);
    int total_columns = pow(kernel,2);
    printf("-------------->im2row kernel: %ld,  padding: %ld, stride: %ld,  in_channels: %ld, output_channels: %ld\n", kernel, padding, stride, channels, output_channels);
  
    vector<Ciphertext> enc_data;
    enc_data.resize(channels*total_rows*total_columns);
    printf("total_rows: %ld, total_columns: %ld, channels: %ld, enc_data size: %ld\n", total_rows, total_columns, channels, enc_data.size());
    // number of rows seen
    auto step = (height_w_padding - kernel + stride)/stride;
    // -----------------------------------------

    auto params=_data.subscript_vector({{0, 1}})[0].parms_id();

    // padding cordinates for comprobation (square matrix)
    int ini_j = padding;
    int final_j = (width + padding - 1);
    int ini_i = padding;
    int final_i = (height + padding - 1);
    // -----------------------------------------
    // parallel task for each core
    task_n3 worker_func = [&](size_t start, size_t end, size_t batch_size) -> bool {

        Ciphertext zero;
        this->tenseal_context()->encrypt_zero(params, zero);
        zero.scale() = this->_init_scale;

        int c = start/total_rows;

        for (size_t row = start; row < end; row++) {  
            int c= row/total_rows;
       

            int start_= row - (c*total_rows);
            int div = start_/step;
            int i = div*stride;
            int j= (start_-(step*div))*stride;

            int resto_t =  row - c*total_rows;  
            int pos_vector=resto_t * channels*total_columns +c*total_columns;

            int patch_ini_i = -1;
            int patch_fin_i = -1;
            int patch_ini_j = -1;
            int patch_fin_j = -1;

            int begin_i = 0;
            int begin_j = 0;

            int end_i = 0;
            int end_j = 0;


            if (i>=ini_i) {
                patch_ini_i=i;
                if (i+kernel<=final_i)
                    patch_fin_i=i+kernel;
                else{
                    patch_fin_i=final_i + 1;   
                    end_i= (i + kernel - 1 - final_i)*kernel;
                }
                          
            }
            else if (i+kernel >= ini_i){
                    begin_i= (ini_i - i)*kernel;
                    patch_ini_i=ini_i;
                    if (final_i>=i+kernel)
                        patch_fin_i=i+kernel;
                    else{
                        patch_fin_i=final_i + 1;   
                        end_i= (i + kernel- 1 - final_i)*kernel;
                }
            }

            if (j>=ini_j) {
                patch_ini_j=j;
                if (j+kernel<=final_j)
                    patch_fin_j=j+kernel;
                 else{
                    patch_fin_j=final_j + 1;   
                    end_j= j + kernel - final_j -1;
                }      
            }
            else if (j+kernel >= ini_j){
                    begin_j= ini_j - j;
                    patch_ini_j=ini_j;
                    if (final_j>=j+kernel)
                        patch_fin_j=j+kernel;
                    else{
                        patch_fin_j=final_j + 1;   
                        end_j= j + kernel - final_j -1 ;
                } 
            }
           
            if (patch_ini_i!=-1 and patch_ini_j!=-1){
                auto access_ini_i= patch_ini_i - padding;
                auto access_ini_j= patch_ini_j - padding;
                auto access_fin_i= patch_fin_i - padding;
                auto access_fin_j= patch_fin_j - padding;
              
                int pair_1 = (access_ini_i++)*height + access_ini_j;
                int pair_2 = pair_1 + (access_fin_j - access_ini_j);
                int pairs_dist=pair_2-pair_1;
           
                vector<Ciphertext> vec;
                if (channels>1) {
                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2),pair<size_t, size_t>(c, c+1)});
                }
                else{
                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2)});
                }




                for (size_t pos=0; pos< begin_i; ++pos) {
         
                    enc_data[pos_vector]= zero;
                    pos_vector++;
                }
                int cont_begin_j=0;
                int cont_end_j=0;
                int cont_vec=0;
                int vec_size=kernel-begin_j-end_j;
                int cont_pair=0;
               
                int pos_vec=0;
                int end_while = pos_vector + total_columns - (begin_i + end_i);
                while(pos_vector < end_while) {
                        if (cont_begin_j<begin_j){
                            
                            enc_data[pos_vector++]= zero;
                            cont_begin_j++;
                        }
                        else if (cont_vec!=vec_size) {
                           
                            enc_data[pos_vector++]= vec[cont_pair++];
                            cont_vec++;
                            if (cont_pair == pairs_dist){
                                pair_1 = (access_ini_i++)*height + access_ini_j;
                                pair_2 = pair_1 + (access_fin_j - access_ini_j);
                                pairs_dist=pair_2-pair_1;
                                
                                if (channels>1) {
                                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2),pair<size_t, size_t>(c, c+1)});
                                }
                                else{
                                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2)});
                                }
                               
                                cont_pair=0;
                            }
                        }
                        else {
                            if (cont_end_j<end_j){
                               
                                enc_data[pos_vector++]= zero;
                                cont_end_j++;
                                if (! (cont_end_j<end_j)){
                                    cont_begin_j=0;
                                    cont_vec=0;
                                    cont_end_j=0;
                                }
                            } else {
                           
                                cont_begin_j=0;
                                cont_vec=0;
                                    
                            }
                            }
                }
                //add zeros end_i     
                for (size_t pos=0; pos<end_i; ++pos) {
                    
                    enc_data[pos_vector]= zero;
                    pos_vector++;
                }
               
            }     
            else { //all zeros
                for (size_t pos=0; pos< total_columns; ++pos) {
                        
                        enc_data[pos_vector]= zero;
                        pos_vector++;
                    }
                    

        }}
        return true;
    };
    
    this->dispatch_jobs_im2row_channel(worker_func, height_w_padding, kernel, stride, channels);
    
    clock_t start= clock();
    CKKSTensor newTensor = CKKSTensor(shared_from_this(), enc_data, {total_rows, total_columns*channels});
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f\n", time_taken);
   
    return newTensor;
}

/*------------------------------------------------------------------------*/ //IMROW

shared_ptr<CKKSTensor>  CKKSTensor::im2row_(const int kernel, const int stride, const int channels, const int padding, const int output_channels) {
    auto this_shape = this->shape(); 
    auto original_height = this_shape[0];
    // -----------------------------------------
    int height=sqrt(original_height);
    printf("height: %ld, original_height: %ld\n", height, original_height);
    int width = height;
    // -----------------------------------------

    int height_w_padding = height  + 2*padding;
    int total_rows =pow(((height_w_padding - kernel + stride)/stride), 2);
    int total_columns = pow(kernel,2);
    printf("-------------->v3 kernel: %ld,  padding: %ld, stride: %ld,  in_channels: %ld, output_channels: %ld\n", kernel, padding, stride, channels, output_channels);

    vector<Ciphertext> enc_data;
    enc_data.resize(channels*total_rows*total_columns);
    printf("total_rows: %ld, total_columns: %ld, channels: %ld, enc_data size: %ld\n", total_rows, total_columns, channels, enc_data.size());
    // number of rows seen
    auto step = (height_w_padding - kernel + stride)/stride;
    // -----------------------------------------
   

    auto params=_data.subscript_vector({{0, 1}})[0].parms_id();

    // padding cordinates for comprobation (square matrix)
    int ini_j = padding;
    int final_j = (width + padding - 1);
    int ini_i = padding;
    int final_i = (height + padding - 1);
    // -----------------------------------------
    // parallel task for each core
    task_n3 worker_func = [&](size_t start, size_t end, size_t batch_size) -> bool {

        Ciphertext zero;
        this->tenseal_context()->encrypt_zero(params, zero);
        zero.scale() = this->_init_scale;

        int c = start/total_rows;
     
        for (size_t row = start; row < end; row++) {  
            int c= row/total_rows;
          
            int start_= row - (c*total_rows);
            int div = start_/step;
            int i = div*stride;
            int j= (start_-(step*div))*stride;

            int resto_t =  row - c*total_rows;  
            int pos_vector=resto_t * channels*total_columns +c*total_columns;

            int patch_ini_i = -1;
            int patch_fin_i = -1;
            int patch_ini_j = -1;
            int patch_fin_j = -1;

            int begin_i = 0;
            int begin_j = 0;

            int end_i = 0;
            int end_j = 0;


            if (i>=ini_i) {
                patch_ini_i=i;
                if (i+kernel<=final_i)
                    patch_fin_i=i+kernel;
                else{
                    patch_fin_i=final_i + 1;   
                    end_i= (i + kernel - 1 - final_i)*kernel;
                }
                          
            }
            else if (i+kernel >= ini_i){
                    begin_i= (ini_i - i)*kernel;
                    patch_ini_i=ini_i;
                    if (final_i>=i+kernel)
                        patch_fin_i=i+kernel;
                    else{
                        patch_fin_i=final_i + 1;   
                        end_i= (i + kernel- 1 - final_i)*kernel;
                }
            }

            if (j>=ini_j) {
                patch_ini_j=j;
                if (j+kernel<=final_j)
                    patch_fin_j=j+kernel;
                 else{
                    patch_fin_j=final_j + 1;   
                    end_j= j + kernel - final_j -1;
                }      
            }
            else if (j+kernel >= ini_j){
                    begin_j= ini_j - j;
                    patch_ini_j=ini_j;
                    if (final_j>=j+kernel)
                        patch_fin_j=j+kernel;
                    else{
                        patch_fin_j=final_j + 1;   
                        end_j= j + kernel - final_j -1 ;
                } 
            }
          
            if (patch_ini_i!=-1 and patch_ini_j!=-1){
                auto access_ini_i= patch_ini_i - padding;
                auto access_ini_j= patch_ini_j - padding;
                auto access_fin_i= patch_fin_i - padding;
                auto access_fin_j= patch_fin_j - padding;
               
                int pair_1 = (access_ini_i++)*height + access_ini_j;
                int pair_2 = pair_1 + (access_fin_j - access_ini_j);
                int pairs_dist=pair_2-pair_1;
              
                vector<Ciphertext> vec;
                if (channels>1) {
                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2),pair<size_t, size_t>(c, c+1)});
                }
                else{
                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2)});
                }
                

                //add zeros begin_i

                for (size_t pos=0; pos< begin_i; ++pos) {
                    
                    enc_data[pos_vector]= zero;
                    pos_vector++;
                }
                int cont_begin_j=0;
                int cont_end_j=0;
                int cont_vec=0;
                int vec_size=kernel-begin_j-end_j;
                int cont_pair=0;
             
                int pos_vec=0;
                int end_while = pos_vector + total_columns - (begin_i + end_i);
                while(pos_vector < end_while) {
                        if (cont_begin_j<begin_j){
                           
                            enc_data[pos_vector++]= zero;
                            cont_begin_j++;
                        }
                        else if (cont_vec!=vec_size) {
                           
                            enc_data[pos_vector++]= vec[cont_pair++];
                            cont_vec++;
                            if (cont_pair == pairs_dist){
                                pair_1 = (access_ini_i++)*height + access_ini_j;
                                pair_2 = pair_1 + (access_fin_j - access_ini_j);
                                pairs_dist=pair_2-pair_1;
                              
                                if (channels>1) {
                                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2),pair<size_t, size_t>(c, c+1)});
                                }
                                else{
                                    vec = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2)});
                                }
                                cont_pair=0;
                            }
                        }
                        else {
                            if (cont_end_j<end_j){
                             
                                enc_data[pos_vector++]= zero;
                                cont_end_j++;
                                if (! (cont_end_j<end_j)){
                                    cont_begin_j=0;
                                    cont_vec=0;
                                    cont_end_j=0;
                                }
                            } else {
                             
                                cont_begin_j=0;
                                cont_vec=0;
                                    
                            }
                            }
                }
                //add zeros end_i     
                for (size_t pos=0; pos<end_i; ++pos) {
                   
                    enc_data[pos_vector]= zero;
                    pos_vector++;
                }
               
            }     
            else { //all zeros
                for (size_t pos=0; pos< total_columns; ++pos) {
                        
                        enc_data[pos_vector]= zero;
                        pos_vector++;
                    }
                   

        }}
        return true;
    };
    
    this->dispatch_jobs_im2row_channel(worker_func, height_w_padding, kernel, stride, channels);
    
    clock_t start= clock();
   
    this->_data = TensorStorage<Ciphertext>(enc_data,  {total_rows, total_columns*channels});
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f\n", time_taken);
    return shared_from_this();
    
}


/*------------------------------------------------------------------------*/ //POOLING



shared_ptr<CKKSTensor>  CKKSTensor::pooling_layer(const int kernel, const int stride, const int channels, const int padding, const int output_channels) {
    auto this_shape = this->shape(); 
    auto original_height = this_shape[0];
    // -----------------------------------------
    int height=sqrt(original_height);
    printf("height: %ld, original_height: %ld\n", height, original_height);
    int width = height;
    // -----------------------------------------
    int height_w_padding = height  + 2*padding;
    int total_rows =pow(((height_w_padding - kernel + stride)/stride), 2);
    //Save only one elemnt (add) of kernel over input
    int total_columns = 1; 
    printf("-------------->pooling layer kernel: %ld,  padding: %ld, stride: %ld,  in_channels: %ld, output_channels: %ld\n", kernel, padding, stride, channels, output_channels);


    vector<Ciphertext> enc_data;
    enc_data.resize(channels*total_rows*total_columns);
    printf("total_rows: %ld, total_columns: %ld, channels: %ld, enc_data size: %ld\n", total_rows, total_columns, channels, enc_data.size());
    // number of rows seen
    auto step = (height_w_padding - kernel + stride)/stride;
    // -----------------------------------------

    auto params=_data.subscript_vector({{0, 1}})[0].parms_id();

    // padding cordinates for comprobation (square matrix)
    int ini_j = padding;
    int final_j = (width + padding - 1);
    int ini_i = padding;
    int final_i = (height + padding - 1);
    // -----------------------------------------
    // parallel task for each core
    task_n3 worker_func = [&](size_t start, size_t end, size_t batch_size) -> bool {

        Ciphertext zero;
        this->tenseal_context()->encrypt_zero(params, zero);
        zero.scale() = this->_init_scale;

        auto evaluator = this->tenseal_context()->evaluator;


        for (size_t row = start; row < end; row++) {  
            int c= row/total_rows;

            int start_= row - (c*total_rows);
            int div = start_/step;
            int i = div*stride;
            int j= (start_-(step*div))*stride;

            int resto_t =  row - c*total_rows;  
            int pos_vector=resto_t * channels*total_columns +c*total_columns;

            int patch_ini_i = -1;
            int patch_fin_i = -1;
            int patch_ini_j = -1;
            int patch_fin_j = -1;

            int begin_i = 0;
            int begin_j = 0;

            int end_i = 0;
            int end_j = 0;


            if (i>=ini_i) {
                patch_ini_i=i;
                if (i+kernel<=final_i)
                    patch_fin_i=i+kernel;
                else{
                    patch_fin_i=final_i + 1;   
                    end_i= (i + kernel - 1 - final_i)*kernel;
                }
                          
            }
            else if (i+kernel >= ini_i){
                    begin_i= (ini_i - i)*kernel;
                    patch_ini_i=ini_i;
                    if (final_i>=i+kernel)
                        patch_fin_i=i+kernel;
                    else{
                        patch_fin_i=final_i + 1;   
                        end_i= (i + kernel- 1 - final_i)*kernel;
                }
            }

            if (j>=ini_j) {
                patch_ini_j=j;
                if (j+kernel<=final_j)
                    patch_fin_j=j+kernel;
                 else{
                    patch_fin_j=final_j + 1;   
                    end_j= j + kernel - final_j -1;
                }      
            }
            else if (j+kernel >= ini_j){
                    begin_j= ini_j - j;
                    patch_ini_j=ini_j;
                    if (final_j>=j+kernel)
                        patch_fin_j=j+kernel;
                    else{
                        patch_fin_j=final_j + 1;   
                        end_j= j + kernel - final_j -1 ;
                } 
            }
            
            
            //Put zero
            enc_data[pos_vector]=zero;

           
            if (patch_ini_i!=-1 and patch_ini_j!=-1){
                auto access_ini_i= patch_ini_i - padding;
                auto access_ini_j= patch_ini_j - padding;
                auto access_fin_i= patch_fin_i - padding;
                auto access_fin_j= patch_fin_j - padding;



                //accum 
                Ciphertext acc(*this->tenseal_context()->seal_context(), params);


                vector<pair<size_t, size_t>> pairs;
                for (size_t pos_i=access_ini_i; pos_i<access_fin_i; ++pos_i) {
                    int pair_1 =pos_i*height + access_ini_j;
                    int pair_2 = pair_1 + (access_fin_j - access_ini_j);
                   
                    vector<Ciphertext> to_sum;
                    if (channels>1) {
                        to_sum = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2),pair<size_t, size_t>(c, c+1)});
                    } else {
                        to_sum = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2)});;
                    }
                    evaluator->add_many(to_sum, acc);
                    this->perform_op(enc_data[pos_vector], acc, OP::ADD);


                    
                }
               }
             
            }
        return true;
    };
    
    this->dispatch_jobs_im2row_channel(worker_func, height_w_padding, kernel, stride, channels);
    
    clock_t start= clock();
 
    this->_data = TensorStorage<Ciphertext>(enc_data,  {total_rows, total_columns*channels});
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f\n", time_taken);
    return shared_from_this();
   
}

/*------------------------------------------------------------------------*/ //POOLING



shared_ptr<CKKSTensor>  CKKSTensor::pooling_layer_(const int kernel, const int stride, const int channels, const int padding, const int output_channels) {
    auto this_shape = this->shape(); 
    auto original_height = this_shape[0];
    // -----------------------------------------
    int height=sqrt(original_height);
    printf("height: %ld, original_height: %ld\n", height, original_height);
    int width = height;
    // -----------------------------------------
    int height_w_padding = height  + 2*padding;
    int total_rows =pow(((height_w_padding - kernel + stride)/stride), 2);
    // Guardamos solo un elemento (suma) del kernel sobre la entrada
    int total_columns = 1; 
    printf("-------------->pooling layer kernel: %ld,  padding: %ld, stride: %ld,  in_channels: %ld, output_channels: %ld\n", kernel, padding, stride, channels, output_channels);


    vector<Ciphertext> enc_data;
    enc_data.resize(channels*total_rows*total_columns);
    printf("total_rows: %ld, total_columns: %ld, channels: %ld, enc_data size: %ld\n", total_rows, total_columns, channels, enc_data.size());
    // number of rows seen
    auto step = (height_w_padding - kernel + stride)/stride;
    // -----------------------------------------

    auto params=_data.subscript_vector({{0, 1}})[0].parms_id();

    // padding cordinates for comprobation (matriz cuadrada)
    int ini_j = padding;
    int final_j = (width + padding - 1);
    int ini_i = padding;
    int final_i = (height + padding - 1);
    // -----------------------------------------
    // parallel task for each core
    task_n3 worker_func = [&](size_t start, size_t end, size_t batch_size) -> bool {

        Ciphertext zero;
        this->tenseal_context()->encrypt_zero(params, zero);
        zero.scale() = this->_init_scale;

        auto evaluator = this->tenseal_context()->evaluator;


        for (size_t row = start; row < end; row++) {  
            int c= row/total_rows;
           

            int start_= row - (c*total_rows);
            int div = start_/step;
            int i = div*stride;
            int j= (start_-(step*div))*stride;

            int resto_t =  row - c*total_rows;  
            int pos_vector=resto_t * channels*total_columns +c*total_columns;

            int patch_ini_i = -1;
            int patch_fin_i = -1;
            int patch_ini_j = -1;
            int patch_fin_j = -1;

            int begin_i = 0;
            int begin_j = 0;

            int end_i = 0;
            int end_j = 0;


            if (i>=ini_i) {
                patch_ini_i=i;
                if (i+kernel<=final_i)
                    patch_fin_i=i+kernel;
                else{
                    patch_fin_i=final_i + 1;   
                    end_i= (i + kernel - 1 - final_i)*kernel;
                }
                          
            }
            else if (i+kernel >= ini_i){
                    begin_i= (ini_i - i)*kernel;
                    patch_ini_i=ini_i;
                    if (final_i>=i+kernel)
                        patch_fin_i=i+kernel;
                    else{
                        patch_fin_i=final_i + 1;   
                        end_i= (i + kernel- 1 - final_i)*kernel;
                }
            }

            if (j>=ini_j) {
                patch_ini_j=j;
                if (j+kernel<=final_j)
                    patch_fin_j=j+kernel;
                 else{
                    patch_fin_j=final_j + 1;   
                    end_j= j + kernel - final_j -1;
                }      
            }
            else if (j+kernel >= ini_j){
                    begin_j= ini_j - j;
                    patch_ini_j=ini_j;
                    if (final_j>=j+kernel)
                        patch_fin_j=j+kernel;
                    else{
                        patch_fin_j=final_j + 1;   
                        end_j= j + kernel - final_j -1 ;
                } 
            }
            
            
            //Put zero
            enc_data[pos_vector]=zero;

           
            if (patch_ini_i!=-1 and patch_ini_j!=-1){
                auto access_ini_i= patch_ini_i - padding;
                auto access_ini_j= patch_ini_j - padding;
                auto access_fin_i= patch_fin_i - padding;
                auto access_fin_j= patch_fin_j - padding;



                //accum add
                Ciphertext acc(*this->tenseal_context()->seal_context(), params);


                vector<pair<size_t, size_t>> pairs;
                for (size_t pos_i=access_ini_i; pos_i<access_fin_i; ++pos_i) {
                    int pair_1 = pos_i*height + access_ini_j;
                    int pair_2 = pair_1 + (access_fin_j - access_ini_j);
                    int dist = pair_2-pair_1;
 
                    vector<Ciphertext> to_sum;
                    if (channels>1) {
                        to_sum = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2),pair<size_t, size_t>(c, c+1)});
                    } else {
                        to_sum = _data.subscript_vector({pair<size_t, size_t>(pair_1, pair_2)});;
                    }
                    printf("pair_1: %ld, pair_2: %ld, c: %ld, dif i: %ld, dif j: %ld, dist: %ld, to_sum: %ld,   pos_vector: %ld \n", pair_1, pair_2, c,  
                    access_fin_i - access_ini_i, access_fin_j - access_ini_j, dist, to_sum.size(), pos_vector);
                    evaluator->add_many(to_sum, acc);
                    this->perform_op(enc_data[pos_vector], acc, OP::ADD);


                    
                }
               }
              
            }
        return true;
    };
    
    this->dispatch_jobs_im2row_channel(worker_func, height_w_padding, kernel, stride, channels);
    
    clock_t start= clock();
    this->_data = TensorStorage<Ciphertext>(enc_data,  {total_rows, total_columns*channels});
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f\n", time_taken);
    return shared_from_this();
   
}

/*-----------------------------------------------------------------------*/ //CONV DIRECT

shared_ptr<CKKSTensor>  CKKSTensor::conv_direct_(const int kernel, const int stride, const int channels, const int padding, const int output_channels, const PlainTensor<double>& other_weight,  const PlainTensor<double>& other_bias) {
    auto this_shape = this->shape(); 
    auto original_height = this_shape[0];
    // -----------------------------------------
    int height=sqrt(original_height);
    printf("height: %ld, original_height: %ld\n", height, original_height);
    int width = height;
    // -----------------------------------------
    int height_w_padding = height  + 2*padding;
    int total_rows =pow(((height_w_padding - kernel + stride)/stride), 2);
    // Salida convolución
    int total_columns = output_channels; 
    printf("-------------->conv_direct_ layer kernel: %ld,  padding: %ld, stride: %ld,  in_channels: %ld, output_channels: %ld\n", kernel, padding, stride, channels, output_channels);


    vector<Ciphertext> enc_data;
    enc_data.resize(total_rows*total_columns);
    printf("total_rows: %ld, total_columns: %ld, channels: %ld, enc_data size: %ld\n", total_rows, total_columns, channels, enc_data.size());
    // number of rows seen
    auto step = (height_w_padding - kernel + stride)/stride;
    // -----------------------------------------

    auto params=  _data.subscript_vector({{0, 1}})[0].parms_id();

    // padding cordinates for comprobation (matriz cuadrada)
    int ini_j = padding;
    int final_j = (width + padding - 1);
    int ini_i = padding;
    int final_i = (height + padding - 1);
    // -----------------------------------------
    // parallel task for each core
    task_n3 worker_func = [&](size_t start, size_t end, size_t batch_size) -> bool {

        Ciphertext zero;
        this->tenseal_context()->encrypt_zero(params, zero);
        zero.scale() = this->_init_scale;

        auto evaluator = this->tenseal_context()->evaluator;
       
      
        for (int pos_vector = start; pos_vector < end; pos_vector++) {  

            // fila en la que estoy del im2row  (varias hebras acceden a la misma)
            int start_= pos_vector/total_columns;
            // pos i j de la fila im2row
            int div = start_/step;
            int i = div*stride;
            int j= (start_-(step*div))*stride;
 
    
            int patch_ini_i = -1;
            int patch_fin_i = -1;
            int patch_ini_j = -1;
            int patch_fin_j = -1;

            int begin_i = 0;
            int begin_j = 0;

            int end_i = 0;
            int end_j = 0;


            if (i>=ini_i) {
                patch_ini_i=i;
                if (i+kernel<=final_i)
                    patch_fin_i=i+kernel;
                else{
                    patch_fin_i=final_i + 1;   
                    end_i= (i + kernel - 1 - final_i)*kernel;
                }
                          
            }
            else if (i+kernel >= ini_i){
                    begin_i= (ini_i - i)*kernel;
                    patch_ini_i=ini_i;
                    if (final_i>=i+kernel)
                        patch_fin_i=i+kernel;
                    else{
                        patch_fin_i=final_i + 1;   
                        end_i= (i + kernel- 1 - final_i)*kernel;
                }
            }

            if (j>=ini_j) {
                patch_ini_j=j;
                if (j+kernel<=final_j)
                    patch_fin_j=j+kernel;
                 else{
                    patch_fin_j=final_j + 1;   
                    end_j= j + kernel - final_j -1;
                }      
            }
            else if (j+kernel >= ini_j){
                    begin_j= ini_j - j;
                    patch_ini_j=ini_j;
                    if (final_j>=j+kernel)
                        patch_fin_j=j+kernel;
                    else{
                        patch_fin_j=final_j + 1;   
                        end_j= j + kernel - final_j -1 ;
                } 
            }
            
            

            size_t pos_other= (pos_vector%total_columns);
            
            //printf("pos_vector: %ld, total_rows: %ld, pos_other: %ld\n", pos_vector, total_rows, pos_other);

            if (patch_ini_i!=-1 and patch_ini_j!=-1){
                auto access_ini_i= patch_ini_i - padding;
                auto access_ini_j= patch_ini_j - padding;
                auto access_fin_i= patch_fin_i - padding;
                auto access_fin_j= patch_fin_j - padding;



                //acumulador suma
                Ciphertext acc(*this->tenseal_context()->seal_context(), params);
                Plaintext pt;
                

                int other_i=begin_i;
                for (size_t pos_i=access_ini_i; pos_i<access_fin_i; ++pos_i) {
                    int pair_1 =pos_i*height + access_ini_j;
                    int pair_2 = pair_1 + (access_fin_j - access_ini_j);

                    //printf("begin_i: %ld, end_i: %ld, begin_j: %ld, end_j: %ld\n", begin_i, end_i, begin_j, end_j);

                    
                    int dist = pair_2 -pair_1;
                

                    //printf("pair_1: %ld , pair_2: %ld, dist: %ld\n ", pair_1, pair_2, dist);
                    for (size_t c=0; c<channels; c++){
                        vector<Ciphertext> to_sum;
                        to_sum.resize(dist);
                        //int col = other_i + begin_j ;
                        int col = (other_i + begin_j)+ c*pow(kernel,2) ;
                        //printf("to_sum: %ld, begin_j: %ld, end_j: %ld, col: %ld\n", to_sum.size(), begin_j, end_j, col);
                        int pos=0;
                        for (size_t d=pair_1; d<pair_2; d++){
                            Plaintext pt;
                            //printf("pos_other: %ld, c: %ld,  col: %ld\n", pos_other, c, col);
                            /*this->tenseal_context()->encode<CKKSEncoder>(
                                other_weight.at({pos_other, c, col++}), pt, this->_init_scale);*/
                            this->tenseal_context()->encode<CKKSEncoder>(
                                    other_weight.at({pos_other, col++}), pt, this->_init_scale);
                            
                            //printf("before mult\n");
                            if (channels>1){
                                to_sum[pos]=this->_data.at({d, c});
                            }
                            else{
                                to_sum[pos]=this->_data.at({d});
                            }
                            this->perform_plain_op(to_sum[pos++], pt, OP::MUL);
                            //printf("after mult\n");

                        }
                        //printf("before evaluator\n");
                        evaluator->add_many(to_sum, acc);
                        //printf("after evaluator\n");
                        if (c==0 and other_i==begin_i){
                            enc_data[pos_vector]=acc;
                        }
                        else{
                            //printf("before perform op\n");
                            this->perform_op(enc_data[pos_vector], acc, OP::ADD); 
                            //printf("after perform op\n");
                        }
                        //printf("pos_vector: %ld\n", pos_vector);
                    }
                    //other_i+=kernel;     
                    other_i+=kernel;     
                }

                //printf("----------------------------------------------------------------------\n");
               }
               else{
                enc_data[pos_vector]=zero;
               }
               Plaintext bias;
               this->tenseal_context()->encode<CKKSEncoder>(
                other_bias.at({pos_other}), bias, this->_init_scale);
               this->perform_plain_op(enc_data[pos_vector], bias, OP::ADD); 
            }
        return true;
    };
    // num jobs= total_rows*total_columns
    this->dispatch_jobs_im2row_channel(worker_func, height_w_padding, kernel, stride, total_columns);
    
    clock_t start= clock();

    this->_data = TensorStorage<Ciphertext>();
    this->_data = TensorStorage<Ciphertext>(enc_data,  {total_rows, total_columns});
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f\n", time_taken);
    return shared_from_this();
   
}

/
/*-----------------------------------------------------------------------*/ //CONV DIRECT

CKKSTensor  CKKSTensor::conv_direct(const int kernel, const int stride, const int channels, const int padding, const int output_channels, const PlainTensor<double>& other_weight,  const PlainTensor<double>& other_bias) {
    auto this_shape = this->shape(); 
    auto original_height = this_shape[0];
    // -----------------------------------------
    int height=sqrt(original_height);
    printf("height: %ld, original_height: %ld\n", height, original_height);
    int width = height;
    // -----------------------------------------
    int height_w_padding = height  + 2*padding;
    int total_rows =pow(((height_w_padding - kernel + stride)/stride), 2);
    // Output convolution
    int total_columns = output_channels; 
    printf("-------------->conv_direct_ layer kernel: %ld,  padding: %ld, stride: %ld,  in_channels: %ld, output_channels: %ld\n", kernel, padding, stride, channels, output_channels);


    vector<Ciphertext> enc_data;
    enc_data.resize(total_rows*total_columns);
    printf("total_rows: %ld, total_columns: %ld, channels: %ld, enc_data size: %ld\n", total_rows, total_columns, channels, enc_data.size());
    // number of rows seen
    auto step = (height_w_padding - kernel + stride)/stride;
    // -----------------------------------------

    auto params=  _data.subscript_vector({{0, 1}})[0].parms_id();

    // padding cordinates for comprobation (square matrix)
    int ini_j = padding;
    int final_j = (width + padding - 1);
    int ini_i = padding;
    int final_i = (height + padding - 1);
    // -----------------------------------------
    // parallel task for each core
    task_n3 worker_func = [&](size_t start, size_t end, size_t batch_size) -> bool {

        Ciphertext zero;
        this->tenseal_context()->encrypt_zero(params, zero);
        zero.scale() = this->_init_scale;

        auto evaluator = this->tenseal_context()->evaluator;
       
      
        for (int pos_vector = start; pos_vector < end; pos_vector++) {  

            // fila en la que estoy del im2row  (varias hebras acceden a la misma)
            int start_= pos_vector/total_columns;
            // pos i j de la fila im2row
            int div = start_/step;
            int i = div*stride;
            int j= (start_-(step*div))*stride;
 
    
            int patch_ini_i = -1;
            int patch_fin_i = -1;
            int patch_ini_j = -1;
            int patch_fin_j = -1;

            int begin_i = 0;
            int begin_j = 0;

            int end_i = 0;
            int end_j = 0;


            if (i>=ini_i) {
                patch_ini_i=i;
                if (i+kernel<=final_i)
                    patch_fin_i=i+kernel;
                else{
                    patch_fin_i=final_i + 1;   
                    end_i= (i + kernel - 1 - final_i)*kernel;
                }
                          
            }
            else if (i+kernel >= ini_i){
                    begin_i= (ini_i - i)*kernel;
                    patch_ini_i=ini_i;
                    if (final_i>=i+kernel)
                        patch_fin_i=i+kernel;
                    else{
                        patch_fin_i=final_i + 1;   
                        end_i= (i + kernel- 1 - final_i)*kernel;
                }
            }

            if (j>=ini_j) {
                patch_ini_j=j;
                if (j+kernel<=final_j)
                    patch_fin_j=j+kernel;
                 else{
                    patch_fin_j=final_j + 1;   
                    end_j= j + kernel - final_j -1;
                }      
            }
            else if (j+kernel >= ini_j){
                    begin_j= ini_j - j;
                    patch_ini_j=ini_j;
                    if (final_j>=j+kernel)
                        patch_fin_j=j+kernel;
                    else{
                        patch_fin_j=final_j + 1;   
                        end_j= j + kernel - final_j -1 ;
                } 
            }
            
            

            size_t pos_other= (pos_vector%total_columns);
            

            if (patch_ini_i!=-1 and patch_ini_j!=-1){
                auto access_ini_i= patch_ini_i - padding;
                auto access_ini_j= patch_ini_j - padding;
                auto access_fin_i= patch_fin_i - padding;
                auto access_fin_j= patch_fin_j - padding;



                //acum add
                Ciphertext acc(*this->tenseal_context()->seal_context(), params);
                Plaintext pt;
                

                int other_i=begin_i;
                for (size_t pos_i=access_ini_i; pos_i<access_fin_i; ++pos_i) {
                    int pair_1 =pos_i*height + access_ini_j;
                    int pair_2 = pair_1 + (access_fin_j - access_ini_j);

                    
                    int dist = pair_2 -pair_1;
                

                    for (size_t c=0; c<channels; c++){
                        vector<Ciphertext> to_sum;
                        to_sum.resize(dist);
                        
                        int col = (other_i + begin_j)+ c*pow(kernel,2) ;
                        
                        int pos=0;
                        for (size_t d=pair_1; d<pair_2; d++){
                            Plaintext pt;
                          
                            this->tenseal_context()->encode<CKKSEncoder>(
                                    other_weight.at({pos_other, col++}), pt, this->_init_scale);
                            
                            
                            if (channels>1){
                                to_sum[pos]=this->_data.at({d, c});
                            }
                            else{
                                to_sum[pos]=this->_data.at({d});
                            }
                            this->perform_plain_op(to_sum[pos++], pt, OP::MUL);
                           
                        }
                       
                        evaluator->add_many(to_sum, acc);
                        if (c==0 and other_i==begin_i){
                            enc_data[pos_vector]=acc;
                        }
                        else{
                            this->perform_op(enc_data[pos_vector], acc, OP::ADD); 
                          
                        }
                       
                    }
                      
                    other_i+=kernel;     
                }

               
               }
               else{
                enc_data[pos_vector]=zero;
               }
               Plaintext bias;
               this->tenseal_context()->encode<CKKSEncoder>(
                other_bias.at({pos_other}), bias, this->_init_scale);
               this->perform_plain_op(enc_data[pos_vector], bias, OP::ADD); 
            }
        return true;
    };
    // num jobs= total_rows*total_columns
    this->dispatch_jobs_im2row_channel(worker_func, height_w_padding, kernel, stride, total_columns);
    clock_t start= clock();
    CKKSTensor newTensor = CKKSTensor(shared_from_this(), enc_data, {total_rows, total_columns});
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    printf("newTensor: %f, %ld\n", time_taken, newTensor.shape()[0]);
    return newTensor;
   
}






/*-------------------------------------------------------------------------*/
void CKKSTensor::erase() {
    this->clear();
    //delete this;
}


/*------------------------------------------------------------------------*/

void CKKSTensor::clear() {
    this->_data = TensorStorage<Ciphertext>();
    this->_batch_size = optional<double>();
    this->_init_scale = 0;
}

void CKKSTensor::load_proto(const CKKSTensorProto& tensor_proto) {
    if (this->tenseal_context() == nullptr) {
        throw invalid_argument("context missing for deserialization");
    }
    this->clear();

    vector<Ciphertext> enc_data;
    vector<size_t> enc_shape;

    for (int idx = 0; idx < tensor_proto.shape_size(); ++idx) {
        enc_shape.push_back(tensor_proto.shape(idx));
    }
    for (int idx = 0; idx < tensor_proto.ciphertexts_size(); ++idx)
        enc_data.push_back(SEALDeserialize<Ciphertext>(
            *this->tenseal_context()->seal_context(),
            tensor_proto.ciphertexts(idx)));
    this->_init_scale = tensor_proto.scale();
    this->_data = TensorStorage<Ciphertext>(enc_data, enc_shape);
    if (tensor_proto.batch_size())
        this->_batch_size = tensor_proto.batch_size();
}

CKKSTensorProto CKKSTensor::save_proto() const {
    CKKSTensorProto buffer;

    for (auto& ct : this->data()) {
        buffer.add_ciphertexts(SEALSerialize<Ciphertext>(ct));
    }
    for (auto& dim : this->shape()) {
        buffer.add_shape(dim);
    }
    buffer.set_scale(this->_init_scale);
    if (this->_batch_size) buffer.set_batch_size(*this->_batch_size);

    return buffer;
}

void CKKSTensor::load(const std::string& tensor_str) {
    if (!this->has_context()) {
        _lazy_buffer = tensor_str;
        return;
    }

    CKKSTensorProto buffer;
    if (!buffer.ParseFromArray(tensor_str.c_str(),
                               static_cast<int>(tensor_str.size()))) {
        throw invalid_argument("failed to parse CKKS tensor stream");
    }
    this->load_proto(buffer);
}

std::string CKKSTensor::save() const {
    if (_lazy_buffer) return _lazy_buffer.value();

    auto buffer = this->save_proto();
    std::string output;
    output.resize(proto_bytes_size(buffer));

    if (!buffer.SerializeToArray((void*)output.c_str(),
                                 static_cast<int>(proto_bytes_size(buffer)))) {
        throw invalid_argument("failed to save CKKS tensor proto");
    }

    return output;
}

shared_ptr<CKKSTensor> CKKSTensor::copy() const {
    if (_lazy_buffer)
        return shared_ptr<CKKSTensor>(new CKKSTensor(_lazy_buffer.value()));
    return shared_ptr<CKKSTensor>(new CKKSTensor(shared_from_this()));
}

shared_ptr<CKKSTensor> CKKSTensor::deepcopy() const {
    if (_lazy_buffer) return this->copy();

    TenSEALContextProto ctx = this->tenseal_context()->save_proto(
        /*save_public_key=*/true, /*save_secret_key=*/true,
        /*save_galois_keys=*/true, /*save_relin_keys=*/true);
    CKKSTensorProto vec = this->save_proto();
    return CKKSTensor::Create(ctx, vec);
}

vector<Ciphertext> CKKSTensor::data() const { return _data.data(); }
/*----------------------------------------------------------------*/
shared_ptr<CKKSTensor> CKKSTensor::reverse() { 
    this->_data.reverse_data();
    return shared_from_this();
}



/*----------------------------------------------------------------*/
vector<size_t> CKKSTensor::shape_with_batch() const {
    if (_batch_size) {
        auto res = _data.shape();
        res.insert(res.begin(), *_batch_size);
        return res;
    }

    return _data.shape();
}
vector<size_t> CKKSTensor::shape() const { return _data.shape(); }

shared_ptr<CKKSTensor> CKKSTensor::reshape(const vector<size_t>& new_shape) {
    return this->copy()->reshape_inplace(new_shape);
}
shared_ptr<CKKSTensor> CKKSTensor::reshape_inplace(
    const vector<size_t>& new_shape) {
    this->_data.reshape_inplace(new_shape);

    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::broadcast(
    const vector<size_t>& other_shape) const {
    return this->copy()->broadcast_inplace(other_shape);
}
shared_ptr<CKKSTensor> CKKSTensor::broadcast_inplace(
    const vector<size_t>& other_shape) {
    this->_data.broadcast_inplace(other_shape);

    return shared_from_this();
}

shared_ptr<CKKSTensor> CKKSTensor::transpose() const {
    return this->copy()->transpose_inplace();
}
shared_ptr<CKKSTensor> CKKSTensor::transpose_inplace() {
    this->_data.transpose_inplace();

    return shared_from_this();
}

double CKKSTensor::scale() const { return _init_scale; }
}  // namespace tenseal
