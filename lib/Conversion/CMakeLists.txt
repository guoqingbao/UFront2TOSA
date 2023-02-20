add_subdirectory(UfrontToTosa)

add_llvm_library(UfrontConversion 
    Passes.cpp

    DEPENDS
    UfrontToTosaConversion
)
