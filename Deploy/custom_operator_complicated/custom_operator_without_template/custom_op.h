#pragma once
#include <iostream>
#include <onnxruntime_cxx_api.h>

struct Input {
    const char* name;
    std::vector<int64_t> dims;
    std::vector<float> values;
};

struct OrtTensorDimensions : std::vector<int64_t> {

    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
        OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

// 算子kernel定义，主要实现compute函数
struct GroupNormKernel {
    private:
    float epsilon_;
    Ort::CustomOpApi ort_;
    
    public:
    GroupNormKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {
        epsilon_ = ort_.KernelInfoGetAttribute<float>(info, "epsilon");
    }

    void Compute(OrtKernelContext* context);
};

// 实现一个customOp继承自customOpBase
struct GroupNormCustomOp : Ort::CustomOpBase<GroupNormCustomOp, GroupNormKernel> {
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new GroupNormKernel(api, info); };
    const char* GetName() const { return "testgroupnorm"; };

    size_t GetInputTypeCount() const { return 4; };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

