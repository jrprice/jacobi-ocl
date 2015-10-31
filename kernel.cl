kernel void jacobi_row_per_wi(const unsigned N,
                              global double *xold,
                              global double *xnew,
                              global double *A,
                              global double *b)
{
  size_t row = get_global_id(0);

  double tmp = 0.0;
  for (unsigned col = 0; col < N; col++)
  {
    if (row != col)
      tmp += A[row*N + col] * xold[col];
  }
  xnew[row] = (b[row] - tmp) / A[row*N + row];
}

kernel void jacobi_row_per_wg(const unsigned N,
                              global double *xold,
                              global double *xnew,
                              global double *A,
                              global double *b,
                              local  double *scratch)
{
  size_t row = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t lsz = get_local_size(0);

  double tmp = 0.0;
  for (unsigned col = lid; col < N; col+=lsz)
  {
    if (row != col)
      tmp += A[row*N + col] * xold[col];
  }

  scratch[lid] = tmp;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (unsigned offset = lsz/2; offset > 0; offset/=2)
  {
    if (lid < offset)
      scratch[lid] += scratch[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  xnew[row] = (b[row] - scratch[0]) / A[row*N + row];
}
