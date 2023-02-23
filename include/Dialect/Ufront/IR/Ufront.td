#ifndef UFRONT_OPS
#define UFRONT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

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

def Ufront_AddOp : Ufront_Op<"add"> {
    let arguments = (ins
        Ufront_Tensor: $lhs,
        Ufront_Tensor: $rhs
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

def Ufront_ReluOp : Ufront_Op<"relu"> {
    let arguments = (ins
        Ufront_Tensor: $input
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
        Ufront_IntArrayAttr2: $padding,
        I64Attr: $groups
    );

    let results = (outs
        Ufront_Tensor4D: $output
    );
}

def Ufront_BatchnormOp : Ufront_Op<"batchnorm"> {
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

def Ufront_SoftmaxOp : Ufront_Op<"softmax"> {
    let arguments = (ins
        Ufront_Tensor: $input
    );

    let results = (outs
        Ufront_Tensor: $output
    );
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

def Ufront_ElidedOp : Ufront_Op<"elided"> {
    let summary = "Placeholder for `tosa.const` that has elided resource";

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

def Ufront_GeluOp : Ufront_Op<"gelu"> {
    let arguments = (ins
        Ufront_Tensor: $input
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
        Ufront_Tensor: $q,
        Ufront_Tensor: $k,
        Ufront_Tensor: $v
    );

    let results = (outs
        Ufront_Tensor: $output
    );
}

#endif // UFRONT_OPS
