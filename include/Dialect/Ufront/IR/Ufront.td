#ifndef UFRONT_OPS
#define UFRONT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def Ufront_Dialect : Dialect {
    let name = "ufront";
    let cppNamespace = "::mlir::ufront";
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
        Ufront_Tensor2D: $input
    );

    let results = (outs
        Ufront_Tensor2D: $output
    );
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

#endif // UFRONT_OPS
