add_subdirectory(UfrontToTosa)
add_subdirectory(TestParameterToConst)

add_llvm_library(UfrontConversion 
    Passes.cpp

    DEPENDS
    UfrontToTosaConversion
    TestParameterToConstConversion
)
