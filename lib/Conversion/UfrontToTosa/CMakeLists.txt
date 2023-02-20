add_mlir_conversion_library(UfrontToTosaConversion
    UfrontToTosa.cpp
    Patterns.cpp
    Util.cpp

    DEPENDS
    UfrontConversionIncGen
)
