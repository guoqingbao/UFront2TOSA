#ifndef UFRONT_OPS
#define UFRONT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Ufront_Dialect : Dialect {
    let name = "ufront";
    let cppNamespace = "::mlir::ufront";

    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Ufront_Op<string mnemonic, list<Trait> traits = []> : Op<Ufront_Dialect, mnemonic, traits>;

def Ufront_IntArrayAttr2 : ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>;
def Ufront_Float : AnyTypeOf<[F16, F32]>;
def Ufront_Tensor : TensorOf<[Ufront_Float]>;
def Ufront_Tensor1D : 1DTensorOf<[Ufront_Float]>;
def Ufront_Tensor2D : 2DTensorOf<[Ufront_Float]>;
def Ufront_Tensor3D : 3DTensorOf<[Ufront_Float]>;
def Ufront_Tensor4D : 4DTensorOf<[Ufront_Float]>;

class Ufront_ActivationOp<string mnemonic, list<Trait> traits = [SameOperandsAndResultType]> : Ufront_Op<mnemonic, traits> {
    let arguments = (ins
        Ufront_Tensor: $input
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

class Ufront_BinaryOp<string mnemonic, list<Trait> traits = []> : Ufront_Op<mnemonic, traits> {
    let arguments = (ins
        Ufront_Tensor: $lhs,
        Ufront_Tensor: $rhs
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_FlatOp : Ufront_Op<"flat"> {
    let arguments = (ins
        Ufront_Tensor: $input
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_Conv2DOp : Ufront_Op<"conv2d"> {
    let arguments = (ins
        Ufront_Tensor4D: $input,
        Ufront_IntArrayAttr2: $kernel,
        Ufront_IntArrayAttr2: $stride,
        Ufront_IntArrayAttr2: $pad,
        I64Attr: $groups
    );

    let results = (outs
        Ufront_Tensor4D: $output
    );
}

def Ufront_BatchNormOp : Ufront_Op<"batchnorm"> {
    let arguments = (ins
        Ufront_Tensor4D: $input
    );

    let results = (outs
        Ufront_Tensor4D: $output
    );
}

def Ufront_LinearOp : Ufront_Op<"linear"> {
    let arguments = (ins
        Ufront_Tensor: $input
    );

    let results = (outs
        Ufront_Tensor: $output
    );

    let hasVerifier = 1;
}

def Ufront_Pool2DOp : Ufront_Op<"pool2d"> {
    let arguments = (ins
        Ufront_Tensor4D: $input,
        StrAttr: $pool_type
    );

    let results = (outs
        Ufront_Tensor4D: $output
    );
}

def Ufront_ParameterOp : Ufront_Op<"parameter"> {
    let results = (outs
        Ufront_Tensor: $tensor
    );

    let builders = [
        OpBuilder<(ins "ArrayRef<int64_t>": $shape, "Type": $element_type)>
    ];
}

def Ufront_ReshapeOp : Ufront_Op<"reshape"> {
    let arguments = (ins
        Ufront_Tensor: $input,
        I64ArrayAttr: $shape
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_TransposeOp : Ufront_Op<"transpose"> {
    let arguments = (ins
        Ufront_Tensor: $input,
        I64ArrayAttr: $perms
    );
    
    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_ExpandOp : Ufront_Op<"expand"> {
    let arguments = (ins
        Ufront_Tensor: $input,
        I64ArrayAttr: $sizes
    );

    let results = (outs
        Ufront_Tensor: $output
    );

    let hasVerifier = 1;
}

def Ufront_ConcatOp : Ufront_Op<"concat"> {
    let arguments = (ins
        Ufront_Tensor: $lhs,
        Ufront_Tensor: $rhs,
        I64Attr: $axis
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_DropoutOp : Ufront_Op<"dropout"> {
    let arguments = (ins
        Ufront_Tensor: $input,
        F64Attr: $rate,
        I64Attr: $seed
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_SliceOp : Ufront_Op<"slice"> {
    let arguments = (ins
        Ufront_Tensor: $input
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_LayerNormOp : Ufront_Op<"layer_norm"> {
    let arguments = (ins
        Ufront_Tensor: $input
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_MultiheadAttentionOp : Ufront_Op<"multihead_attention"> {
    let arguments = (ins
        Ufront_Tensor3D: $query,
        Ufront_Tensor3D: $key,
        Ufront_Tensor3D: $value,
        Optional<Ufront_Tensor3D>: $mask,
        I64Attr: $embed_dim,
        I64Attr: $num_heads,
        F64Attr: $dropout,
        BoolAttr: $batch_first
    );

    let results = (outs
        Ufront_Tensor3D: $output
    );
}

def Ufront_StrueDivOp : Ufront_Op<"struediv"> {
    let arguments = (ins
        Ufront_Tensor: $input,
        F64Attr: $scalar
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_EqOp : Ufront_Op<"eq"> {
    let arguments = (ins
        Ufront_Tensor: $input,
        F64Attr: $comparator
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_MaskedFillOp : Ufront_Op<"masked_fill", [SameOperandsAndResultType]> {
    let arguments = (ins
        Ufront_Tensor: $input,
        Ufront_Tensor: $mask,
        F64Attr: $value
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_ReluOp : Ufront_ActivationOp<"relu">;
def Ufront_SoftmaxOp : Ufront_ActivationOp<"softmax">;
def Ufront_GeluOp : Ufront_ActivationOp<"gelu">;
def Ufront_SiluOp : Ufront_ActivationOp<"silu">;
def Ufront_SigmoidOp : Ufront_ActivationOp<"sigmoid">;
def Ufront_HardSigmoidOp : Ufront_ActivationOp<"hardsigmoid">;
def Ufront_HardSwishOp : Ufront_ActivationOp<"hardswish">;

def Ufront_AddOp : Ufront_BinaryOp<"add">;
def Ufront_MultiplyOp : Ufront_BinaryOp<"multiply">;
def Ufront_BatchMatmulOp : Ufront_BinaryOp<"batch_matmul">;

#endif // UFRONT_OPS
