# Ufront to Tosa Converter

## Setting up

```sh
docker build . -t ufront
```

## Using ufront-opt

```sh
ufront-opt test/ufront.mlir --convert-ufront-to-tosa
```

## Note

`ufront.elided` is placeholder for resource elided `tosa.const`
