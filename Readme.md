e

![lab](/resources/pclogo-2-2.png)
## <div align="center">Lab 05</div>

1. Write a serial and parallel program that uses OpenMP solving issues related to Data Races

## Instructions

1. Use your AWS EC2 instance or another linux instance that has openmp to complete the Lab Exercises
2. For timing the perfornance of your solutons use the code snippet given at the end of the lab sheet

## Exercise 1:
1. Run the code given below in the cluster using the qsub command and briefly describe how the ```pragma omp barrier``` command works in this program.

```
#include <omp.h>
#include <stdio.h>

int main() {

#pragma omp parallel
{
  int th = omp_get_thread_num();
  printf("thread # %d\n", th);

 #pragma omp barrier 
 
 printf("Outside Barrier # %d\n", omp_get_thread_num());
 printf("Hi again from thread # %d\n", omp_get_thread_num());

}

   return 0;
}
```
## Exercise 2:
1. Run the code given below in the AWS EC2 instance and briefly describe how the ```pragma omp single``` command works in this program.
```
#include <omp.h>
#include <stdio.h>

int main() {

#pragma omp parallel
{
  int th = omp_get_thread_num();
  printf("thread # %d\n", th);

 #pragma omp single
 {
    printf("Inside Single # %d\n", omp_get_thread_num());
    printf("Exiting Single \n");
 }
 printf("Hi again from thread # %d\n", omp_get_thread_num());

}
   return 0;
}
```
## Exercise 3:
1. Run the code given below in the AWS EC2 Instance and briefly describe how the ```pragma omp master``` command works in this program.
```
#include <omp.h>
#include <stdio.h>

int main() {

#pragma omp parallel
{
  int th = omp_get_thread_num();
  printf("thread # %d\n", th);
 #pragma omp master 
 {
    printf("Inside Master # %d\n", omp_get_thread_num());
    printf("Exiting Master \n");
 }
 printf("Hi again from thread # %d\n", omp_get_thread_num());
   return 0;
}
```
## Exercise 4:
1. Run the code given below in the cluster using the qsub command  and briefly describe how the threads are generated for this code.
2. When you replace the code below with ```#pragma omp parallel for collapse(2)``` how are the threads generated.

```
#include <omp.h>
#include <stdio.h>

int main() {
 
#pragma omp parallel for 	
  for (int r=1; r<=5; r++)
     for (int k=1; k<=10; k++) {
	 printf("r = %d, k = %d, thread # = %d\n", r, k, omp_get_thread_num());
     }

   return 0;
}
```
## Exercise 5:

Parallelize the following serial code for fibonacci series using task parallelization
```c
 int fib(int n) {
  int i, j;
  if (n<2)
    return n;
  else {
    i=fib(n-1);
    j=fib(n-2);
    return i+j;
  }
}
```
## Exercise 6:

Parallelize the element-wise multiplication of two arrays using OpenMP and the strip mining technique.
Given three arrays A[], B[], and C[], each of size N = 1,000,000, write a C program that computes:

```
C[i] = A[i] × B[i] for all i from 0 to N−1
```
Your solution must:

1. Use strip mining to divide the array into fixed-size strips
2. Use OpenMP to parallelize the computation across multiple threads
3. Define a suitable strip size that aligns with SIMD vector width

## Exercise 7: Area of the Mandelbrot Set

The aim of this exercise is to use the **OpenMP directives** learned so far and apply them to a real problem. It will demonstrate some of the **issues** which need to be taken into account when adapting serial code to a parallel version.

#### The Mandelbrot Set

The **Mandelbrot Set** is the set of complex numbers \( c \) for which the iteration \( z = z^2 + c \) does not diverge, from the initial condition \( z = c \). To determine (approximately) whether a point \( c \) lies in the set, a finite number of iterations are performed, and if the condition \( |z| > 2 \) is satisfied, then the point is considered to be **outside the Set**. What we are interested in is calculating the **area** of the Mandelbrot Set. There is no known theoretical value for this, and estimates are based on a procedure similar to that used here.

#### The Code

The method we shall use generates a grid of points in a box of the complex plane containing the upper half of the (symmetric) **Mandelbrot Set**. Then each point is iterated using the equation \( z = z^2 + c \) a finite number of times (say **2000**). If within that number of iterations the threshold condition \( |z| > 2 \) is satisfied, then that point is considered to be **outside** of the Mandelbrot Set. Then counting the number of points **within** the Set and those **outside** will give an estimate of the **area** of the Set.

Parallelize the serial code using the **OpenMP directives** and **library routines** that you have learned so far. The method for doing this is as follows:

1. Start a parallel region **before** the main loop, ensuring that any **private**, **shared**, or **reduction** variables within the region are correctly declared.
2. Distribute the outermost loop across the threads available so that each thread has an **equal number** of the points. For this, you will need to use some of the **OpenMP library routines**.

Once you have written the code, try it out using **1**, **2**, **3**, and **4** threads. Check that the results are **identical** in each case, and compare the **time taken** for the calculations using the different number of threads.

#### Extra Exercise

Try **different ways** of mapping iterations to threads.

Use the Code Provided in the zip file

## Exercise 6: Molecular Dynamics

The aim of this exercise is to demonstrate how to use **OpenMP critical directives** to parallelize a **molecular dynamics code**.

#### The Code

The code can be found in `*/MolDyn/`. The code is a **molecular dynamics (MD)** simulation of **argon atoms** in a box with **periodic boundary conditions**. The atoms are initially arranged as a **face-centered cubic (FCC) lattice** and then allowed to melt. The interaction of the particles is calculated using a **Lennard-Jones potential**. The main loop of the program is in the file `main.[f|c|f90]`. Once the lattice has been generated and the forces and velocities initialized, the main loop begins. The following steps are undertaken in each iteration of this loop:

1. The particles are moved based on their **velocities**, and the velocities are partially updated (call to `domove`).
2. The **forces** on the particles in their new positions are calculated, and the **virial** and **potential energies** accumulated (call to `forces`).
3. The forces are **scaled**, the velocity update is completed, and the **kinetic energy** calculated (call to `mkekin`).
4. The **average particle velocity** is calculated, and the **temperature** scaled (call to `velavg`).
5. The full **potential** and **virial energies** are calculated and printed out (call to `prnout`).

#### Parallelization

The parallelization of this code is a little less straightforward. There are several **dependencies** within the program which will require use of the `CRITICAL`/`critical` directive as well as the `REDUCTION`/`reduction` clause. The instructions for parallelizing the code are as follows:

1. Edit the subroutine/function `forces.[f|c|f90]`. Start a `!$OMP PARALLEL DO`/`#pragma omp parallel for` for the outer loop in this subroutine, identifying any **private** or **reduction** variables. **Hint**: There are **2 reduction variables**.
2. Identify the variable within the loop which must be updated **atomically** and use `!$OMP CRITICAL`/`#pragma omp critical` to ensure this is the case.

Once this is done, the code should be ready to run in **parallel**. Compare the output using **2**, **3**, and **4** threads with the **serial output** to check that it is working. Try adding the schedule clause with the option `static,n` to the `DO`/`for` directive for different values of \( n \). Does this have any effect on **performance**?

Use the Code Provided in the zip file

## Timing Code

Time your parallel Results
Timing can be obtained as follows
```c
double tstart, tcalc
tstart = omp_get_wtime();

 // code to be timed in here

tstop  = omp_get_wtime();

tcalc = tstop - tstart;  // time in milli seconds







