/*
 ============================================================================
 Name        : matriz_x_vector2.cpp
 Author      : Jose Miguel Mantas Ruiz
 Copyright   : GNU Open Souce and Free license
 Description : Tutorial 5. Multiplicacion de Matrix por Vector.
 ============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <cmath>

using namespace std;

int main(int argc, char *argv[])
{

    int numeroProcesadores, id_Proceso;

    float *A,     // Matriz global a multiplicar
        *x,       // Vector a multiplicar
        *y;       // Vector resultado

    double tInicio, // Tiempo en el que comienza la ejecucion
        Tpar, Tseq;

    if (argc != 2)
    { 
        cout << "The dimension N of the matrix is missing (N x N matrix)" << endl;
        exit(-1);
    }

    int n = atoi(argv[1]);

    // Inicializamos MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_Proceso);
    MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);

    int raiz = sqrt(numeroProcesadores);
    int tam = n / raiz;

    // Definimos los colores
    int row_color = id_Proceso / raiz;
    int col_color = id_Proceso % raiz;
    int diag_color = (row_color == col_color) ? 0 : MPI_UNDEFINED;

    // Definimos los comunicadores
    MPI_Comm row_comm, col_comm, diag_comm;

    // Inicializamos los nuevos comunicadores
    MPI_Comm_split(MPI_COMM_WORLD, row_color, id_Proceso, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, id_Proceso, &col_comm);
    MPI_Comm_split(MPI_COMM_WORLD, diag_color, id_Proceso, &diag_comm);

    // Definimos los nuevos rank y size de los nuevos comunicadores
    int row_rank, row_size;
    int col_rank, col_size;
    int diag_rank, diag_size;

    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    if (diag_color == 0) {
        MPI_Comm_rank(diag_comm, &diag_rank);
        MPI_Comm_size(diag_comm, &diag_size);
    }


    float *buf_envio = new float[n * n];

    // Proceso 0 genera matriz A y vector x
    if (id_Proceso == 0)
    {
        A = new float[n * n]; // reservamos espacio para la matriz (n x n floats)
        x = new float[n];     // reservamos espacio para el vector x (n floats).
        y = new float[n];     // reservamos espacio para el vector resultado final y (n floats)

        // Rellena la matriz y el vector
        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(1.5 * (1 + (5 * (i)) % 3) / (1 + (i) % 5));
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = (float)(1.5 * (1 + (5 * (i + j)) % 3) / (1 + (i + j) % 5));
            }
        }

        // Definimos MPI_BLOQUE
        MPI_Datatype MPI_BLOQUE;
        MPI_Type_vector(tam, tam, n, MPI_FLOAT, &MPI_BLOQUE);
        MPI_Type_commit(&MPI_BLOQUE);

        // Hacemos el MPI_PACK
        for (int i = 0, posicion = 0, comienzo; i < numeroProcesadores; i++) {
            row_size = i / raiz;
            col_size = i % raiz;
            comienzo = (col_size * tam) + (row_size * tam * tam * raiz);
            MPI_Pack(&A[comienzo], 1, MPI_BLOQUE, buf_envio, sizeof(float) * n * n, &posicion, MPI_COMM_WORLD);
        }

        // Liberamos MPI_BLOQUE
        MPI_Type_free(&MPI_BLOQUE);
    }

    // Scatter de los bloques
    float *buf_recep = new float[tam * tam];
    MPI_Scatter(buf_envio, sizeof(float) * tam * tam, MPI_PACKED, buf_recep, tam * tam, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Definimos los vectores local_x, local_y
    float *local_x = new float[tam];
    float *local_y = new float[tam];

    // Scatter de local_x en la diagonal y broacast en las columnas
    if (diag_color == 0)
        MPI_Scatter(x, tam, MPI_FLOAT, local_x, tam, MPI_FLOAT, 0, diag_comm);
    
    MPI_Bcast(local_x, tam, MPI_FLOAT, row_rank, col_comm);

    // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
    // a la vez, para tener mejor control del tiempo empleado
    MPI_Barrier(MPI_COMM_WORLD);

    // Inicio de medicion de tiempo
    tInicio = MPI_Wtime();

    for (int i = 0; i < tam; i++)
    {
        local_y[i] = 0.0;

        // Este bucle for ya no llega hasta n sino hasta tam debido al enfoque por bloques
        for (int j = 0; j < tam; j++)
        {
            local_y[i] += buf_recep[i * tam + j] * local_x[j];
        }
    }

    // Barrera para esperar que terminen todas las hebras la fase de computo
    MPI_Barrier(MPI_COMM_WORLD);

    // fin de medicion de tiempo
    Tpar = MPI_Wtime() - tInicio;

    // El contenido de local_y no es el resultado final, sino que una parte de este
    // Para obtener el resultado final hay que sumar los local_y de todos los procesos
    // Esta operación se lleva a cabo con un MPI_Reduce
    float *reduce_y = new float[tam];
    MPI_Reduce(local_y, reduce_y, tam, MPI_FLOAT, MPI_SUM, col_rank, row_comm);

    // Recogemos los datos de la multiplicacion, por cada proceso sera un escalar
    // y se recoge en un vector, Gather se asegura de que la recolecci�n se haga
    // en el mismo orden en el que se hace el Scatter, con lo que cada escalar
    // acaba en su posicion correspondiente del vector.
    if (diag_color == 0)
        MPI_Gather(reduce_y,         // Dato que envia cada proceso
                tam,    // Numero de elementos que se envian
                MPI_FLOAT,       // Tipo del dato que se envia
                y,               // Vector en el que se recolectan los datos
                tam,    // Numero de datos que se esperan recibir por cada proceso
                MPI_FLOAT,       // Tipo del dato que se recibira
                0,               // proceso que va a recibir los datos
                diag_comm); // Canal de comunicacion (Comunicador Global)

    // Terminamos la ejecucion de los procesos, despues de esto solo existira
    // el proceso 0
    // Ojo! Esto no significa que los demas procesos no ejecuten el resto
    // de codigo despues de "Finalize", es conveniente asegurarnos con una
    // condicion si vamos a ejecutar mas codigo (Por ejemplo, con "if(rank==0)".
    MPI_Finalize();

    if (id_Proceso == 0)
    {
        // Umbral para detectar errores de precisión en el reduce
        float umbral = 0.001f;

        float *comprueba = new float[n];
        // Calculamos la multiplicacion secuencial para
        // despues comprobar que es correcta la solucion.

        tInicio = MPI_Wtime();
        for (int i = 0; i < n; i++)
        {
            comprueba[i] = 0;
            for (int j = 0; j < n; j++)
            {
                comprueba[i] += A[i * n + j] * x[j];
            }
        }
        Tseq = MPI_Wtime() - tInicio;

        int errores = 0;
        for (unsigned int i = 0; i < n; i++)
        {
            cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
            if (comprueba[i] < (y[i] - umbral) && (y[i] + umbral) < comprueba[i])
                errores++;
        }
        cout << ".......Obtained and expected result can be seen above......." << endl;

        delete[] y;
        delete[] comprueba;
        delete[] A;

        if (errores)
        {
            cout << "Found " << errores << " Errors!!!" << endl;
        }
        else
        {
            cout << "No Errors!" << endl
                 << endl;
            cout << "...Parallel time= " << Tpar << " seconds." << endl
                 << endl;
            cout << "...Sequential time= " << Tseq << " seconds." << endl
                 << endl;
        }
    }

    return 0;
}
