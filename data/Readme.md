# Aura Project

## Embeddings

Este proyecto utiliza embeddings preentrenados de FastText en español.

Debido a su tamaño, algunos archivos NO están incluidos en el repositorio.

---

## Archivos incluidos

- data/cc.es.300.vec.bin  (~60MB)

Este archivo ya está procesado y listo para usarse directamente.

---

## Archivos NO incluidos

Los siguientes archivos no se incluyen por su gran tamaño:

- data/cc.es.300.vec (~4.5GB)
- data/cc.es.300.vec.bin.vectors.npy (~2.4GB)

---

## Descarga del dataset original

Puedes descargar los embeddings originales desde:

https://fasttext.cc/docs/en/crawl-vectors.html

Archivo:

cc.es.300.vec.gz

---

## Instalación

1. Descargar el archivo:

   cc.es.300.vec.gz

2. Descomprimir:

   gunzip cc.es.300.vec.gz

3. Mover a la carpeta data:

   data/cc.es.300.vec

---

## Generación de archivos derivados

Si necesitas regenerar los archivos:

- .bin
- .npy

ejecuta los scripts del proyecto correspondientes.

---

## Nota

Los archivos grandes están excluidos mediante .gitignore para mantener el repositorio ligero y manejable.
