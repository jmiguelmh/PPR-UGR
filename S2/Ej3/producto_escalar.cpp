#include "mpi.h"
#include <math.h>
#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]) {
    int tama, rank, size;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    if (argc < 2) {
        if (rank == 0) {
            cout << "No se ha especificado numero de elementos, multiplo de la cantidad de entrada, por defecto sera " << size * 100;
            cout << "\nUso: <ejecutable> <cantidad>" << endl;
        }
        tama = size * 100;
    } else {
        tama = atoi(argv[1]);
        if (tama < size) tama = size;
        else {
            int i = 1, num = size;
            while (tama > num) {
                ++i;
                num = size*i;
            }
            if (tama != num) {
                if (rank == 0)
                    cout << "Cantidad cambiada a " << num << endl;
                tama = num;
            }
        }
    }
 
    // Creacion y relleno de los vectores
    vector<long> VectorA, VectorB, VectorALocal, VectorBLocal;
    VectorA.resize(tama, 0);
    VectorB.resize(tama, 0);
    VectorALocal.resize(tama/size, 0);
    VectorBLocal.resize(tama/size, 0);
    if (rank == 0) {
        for (long i = 0; i < tama; ++i) {
            VectorA[i] = i + 1; // Vector A recibe valores 1, 2, 3, ..., tama
        }
    }
 
    // Repartimos los valores del vector A
    MPI_Scatter(&VectorA[0], // Valores a compartir
            tama / size, // Cantidad que se envia a cada proceso
            MPI_LONG, // Tipo del dato que se enviara
            &VectorALocal[0], // Variable donde recibir los datos
            tama / size, // Cantidad que recibe cada proceso
            MPI_LONG, // Tipo del dato que se recibira
            0,  // proceso principal que reparte los datos
            MPI_COMM_WORLD); // Comunicador (En este caso, el global)
    
    // Damos valores al bloque local del vector B
    int bsize = tama/size;
    int istart = bsize * rank;
    int iend = istart + bsize;

    for (long i = istart; i < iend; i++)
        VectorBLocal[i % bsize] = (i + 1)*10;
    
    // Calculo de la multiplicacion escalar entre vectores
    long producto = 0;
    for (long i = 0; i < tama / size; ++i) {
        producto += VectorALocal[i] * VectorBLocal[i];
    }
    long total;
 
    // Reunimos los datos en un solo proceso, aplicando una operacion
    // aritmetica, en este caso, la suma.
    MPI_Reduce(&producto, // Elemento a enviar
            &total, // Variable donde se almacena la reunion de los datos
            1, // Cantidad de datos a reunir
            MPI_LONG, // Tipo del dato que se reunira
            MPI_SUM, // Operacion aritmetica a aplicar
            0, // Proceso que recibira los datos
            MPI_COMM_WORLD); // Comunicador
 
    if (rank == 0)
        cout << "Total = " << total << endl;
 
	// Terminamos la ejecucion de los procesos, despues de esto solo existira
	// el proceso 0
	// ¡Ojo! Esto no significa que los demas procesos no ejecuten el resto
	// de codigo despues de "Finalize", es conveniente asegurarnos con una
	// condicion si vamos a ejecutar mas codigo (Por ejemplo, con "if(rank==0)".
    MPI_Finalize();
    return 0;
}