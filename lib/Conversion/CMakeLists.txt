add_subdirectory(UfrontToTosa)
add_subdirectory(TestElidedToConst)

add_mlir_conversion_library(FixTosaMatmul
    FixTosaMatmul.cpp
)

add_llvm_library(UfrontConversion 
    Passes.cpp

    DEPENDS
    UfrontToTosaConversion
    TestElidedToConstConversion
    FixTosaMatmul
)
