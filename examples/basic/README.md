# Пример-потребитель ruCLIP

Показывает, что стороннему приложению достаточно `find_package(ruCLIP)` и одной
зависимости `ruclip::ruclip` — без знания путей к исходникам библиотеки.

## Сборка

1. Соберите и установите библиотеку (из корня `RuCLIP/`):

   ```bat
   cmake -S . -B build_lib -G "Visual Studio 17 2022" -A x64 ^
     -DRUCLIP_WITH_TORCH=ON -DRUCLIP_BUILD_CLI=OFF ^
     -DCMAKE_INSTALL_PREFIX=C:/work/clip/ruclip-install
   cmake --build build_lib --config Release
   cmake --install build_lib --config Release
   ```

2. Соберите пример, указав префикс установки:

   ```bat
   cmake -S examples/basic -B build_example -G "Visual Studio 17 2022" -A x64 ^
     -DruCLIP_DIR=C:/work/clip/ruclip-install/lib/cmake/ruCLIP
   cmake --build build_example --config Release
   ```

## Запуск

```bat
set RUCLIP_BACKEND=trt
set RUCLIP_FP16=1
set RUCLIP_MAX_BATCH=8
build_example\Release\ruclip_basic.exe ^
  C:\work\clip\ruclip\CLIP\data\ruclip-vit-large-patch14-336 ^
  C:\work\clip\ruclip\CLIP\data\test_images\1.png ^
  cat bear fox
```

Для TensorRT engine-файл создаётся рядом с ONNX при первом запуске:
`clip_visual.fp16.gpu0.b8.engine` и `clip_textual.fp16.gpu0.b8.engine`
(точность/GPU/maxBatch берутся из настроек).
