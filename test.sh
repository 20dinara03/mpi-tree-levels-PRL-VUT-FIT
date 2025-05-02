#!/bin/bash

# Kontrola vstupního argumentu
if [ "$#" -ne 1 ]; then
    echo "Použití: ./test.sh <tree_string>" >&2
    exit 1
fi

TREE_STR=$1
N=${#TREE_STR}                     # počet uzlů = délka řetězce
NUM_PROCS=$((2 * N - 1))          # počet procesů = 2n - 1
EXECUTABLE="vuv"
SOURCE="vuv.cpp"

# Překlad programu pokud není přeložen (stdout potlačen, stderr zachován)
if [ ! -f "$EXECUTABLE" ]; then
    mpic++ -std=c++17 -o "$EXECUTABLE" "$SOURCE" > /dev/null
    if [ $? -ne 0 ]; then
        echo "Chyba při překladu programu." >&2
        exit 2
    fi
fi

# Spuštění programu s MPI
mpirun --oversubscribe -n "$NUM_PROCS" ./"$EXECUTABLE" "$TREE_STR"
