#include "mpi.h"
#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv); //iniciamos el entorno MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //obtenemos el identificador del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size); //obtenemos el numero de procesos

    MPI_Comm comm // nuevo comunicador para pares o impares
            , comm_inverso; // nuevo para todos los procesos pero con rank inverso.
    int rank_inverso, size_inverso;
    int rank_nuevo, size_nuevo;
    int a;
    int b;

    if (rank == 0) {
        a = 2000;
        b = 1;
    } else {
        a = 0;
        b = 0;
    }

    int color = rank % 2;
    // creamos un nuevo cominicador
    MPI_Comm_split(MPI_COMM_WORLD // a partir del comunicador global.
            , color // los del mismo color entraran en el mismo comunicador
            // lo pares tiene color 0 y los impares 1.
            , rank, // indica el orden de asignacion de rango dentro de los nuevos comunicadores
            &comm); // Referencia al nuevo comunicador creado.
    // creamos un nuevo comunicador inverso.
    MPI_Comm_split(MPI_COMM_WORLD, // a partir del comunicador global.
            0 // el color es el mismo para todos.
            , -rank // el orden de asignacion para el nuevo rango es el inverso al actual.
            , &comm_inverso); // Referencia al nuevo comunicador creado.

    MPI_Comm_rank(comm, &rank_nuevo); // obtenemos el nuevo rango asignado dentro de comm
    MPI_Comm_size(comm, &size_nuevo); // obtenemos numero de procesos dentro del comunicador

    MPI_Comm_rank(comm_inverso, &rank_inverso); // obtenemos el nuevo rango asignado en comm_inverso
    MPI_Comm_size(comm_inverso, &size_inverso); // obtenemos numero de procesos dentro del comunicador

    //Probamos a enviar datos por distintos comunicadores
    MPI_Bcast(&b, 1, MPI_INT,
            size - 1, // el proceso con rango 0 dentro de MPI_COMM_WORLD sera root
            comm_inverso);
    if (color==0) // Sólo para los pares
        MPI_Bcast(&a, 1, MPI_INT,
            0, // el proceso con rango 0 dentro de comm sera root
            comm);
	else a=0;

    // Elemento a recibir del scatter
    int scatterElement = 0;
    vector<int> scatterVector;
    scatterVector.resize(size_nuevo);

    // Si mi rank es impar
    if (rank % 2 == 1) {
        // Primer elemento del comunicador de impares (P1 en MPI_COMM_WORLD)
        if (rank_nuevo == 0) {
            // Inicializamos el vector
            scatterVector.resize(size_nuevo);
            for (int i = 0; i < scatterVector.size(); i++)
                scatterVector[i] = i;
            
        }

        // Operacion scatter
        MPI_Scatter(&scatterVector[0],
                    1,
                    MPI_INT,
                    &scatterElement,
                    1,
                    MPI_INT,
                    0,
                    comm);


        cout << "Soy el proceso " << rank_nuevo << " de " << size_nuevo << " dentro del comunicador impar" << endl;
        cout << "Soy el proceso " << rank << " de " << size << " dentro del comunicador MPI_COMM_WORLD" << endl;
        cout << "He recibido " << scatterElement << " de la operación MPI_Scatter" << endl;
        cout << endl;
    }



    MPI_Finalize();
    return 0;
}