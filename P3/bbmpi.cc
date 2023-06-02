/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

unsigned int NCIUDADES;
int rank, size, siguiente, anterior;

MPI_Comm comunicadorCarga;

int main(int argc, char **argv)
{
    switch (argc)
    {
    case 3:
        NCIUDADES = atoi(argv[1]);
        break;
    default:
        std::cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << std::endl;
        exit(1);
        break;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Colores de los comunicadores
    int carga = 0;
    int cota = 1;

    MPI_Comm_split(MPI_COMM_WORLD, carga, rank, &comunicadorCarga);

    siguiente = (rank + 1 ) % size;
    anterior = (rank - 1 + size) % size;

    int **tsp0 = reservarMatrizCuadrada(NCIUDADES);
    tNodo nodo,   // nodo a explorar
        lnodo,    // hijo izquierdo
        rnodo,    // hijo derecho
        solucion; // mejor solucion
    bool nueva_U,  // hay nuevo valor de c.s.
        fin;
    int U;        // valor de c.s.
    int iteraciones = 0;
    tPila pila; // pila de nodos a explorar

    U = INFINITO;    // inicializa cota superior
    InicNodo(&nodo); // inicializa estructura nodo

    if(rank == 0) {
        LeerMatriz(argv[2], tsp0);
    }
    
    MPI_Bcast(&tsp0[0][0], NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    if(rank != 0) {
        EquilibrarCarga(pila, fin);
        if(!fin)
            pila.pop(nodo);
    }

    fin = Inconsistente(tsp0);

    while(!fin) {
        Ramifica(&nodo, &lnodo, &rnodo, tsp0);
        nueva_U = false;

        if(Solucion(&rnodo)) {
            if(rnodo.ci() < U) {
                U = rnodo.ci();
                nueva_U = true;
            }
        } else {
            if(rnodo.ci() < U)
                pila.push(rnodo);
        }

        if(Solucion(&lnodo)) {
            if(lnodo.ci() < U) {
                U = lnodo.ci();
                nueva_U = true;
            }
        } else {
            if(lnodo.ci() < U)
                pila.push(lnodo);
        }

        if(nueva_U)
            pila.acotar(U);
        
        EquilibrarCarga(pila, fin);

        if(!fin)
            pila.pop(nodo);
        
        iteraciones++;
        std::cout << "Proceso " << rank << ", iteracion: " << iteraciones << ", ci: " << U << std::endl;

    }

    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime() - t;


    MPI_Finalize();

    return 0;
}
