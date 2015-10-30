kernel void jacobi(global double *A,
                   global double *b,
                   global double *xold,
                   global double *xnew,
                   const unsigned N)
{
  size_t i = get_global_id(0);

  double tmp = 0.0;
  for (unsigned j = 0; j < N; j++)
  {
    if (i != j)
      tmp += A[i*N + j] * xold[j];
  }
  xnew[i] = (b[i] - tmp) / A[i*N + i];
}
