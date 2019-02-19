//Run clang 3.8 to generate llvm

clang -S -emit-llvm -o filename.ll -x cl openclfilename.cl

//Run the parser
gcc clllvmparse.c -o parse

//Run executable
./parse filename.ll

//Debug
./parse -v filename.ll

//Read results
cat parsed
