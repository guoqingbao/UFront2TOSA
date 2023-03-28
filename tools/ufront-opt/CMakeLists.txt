get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        cnpy
)

add_llvm_executable(ufront-opt ufront-opt.cpp)
target_link_libraries(ufront-opt PRIVATE ${LIBS})
