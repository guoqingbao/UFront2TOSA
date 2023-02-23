# Ufront to Tosa Converter

## 构建

```sh
docker build . -t ufront
```

## 使用

```sh
ufront-opt test/ufront.mlir --convert-ufront-to-tosa
```

## WIP

`ufront.elided` 暂时用以替代具有 elided resource 的 `tosa.const`，即

```mlir
%1 = "ufront.elided"() : () -> tensor<64x7x7x3xf32>
```

等价于

```mlir
%1 = "tosa.const"() : {value = dense_resource<__elided__> : tensor<64x7x7x3xf32>} : () -> tensor<64x7x7x3xf32>
```
