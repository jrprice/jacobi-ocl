#if USE_MAD24
#define INDEX(col,row,N) mad24(row, N, col)
#else
#define INDEX(col,row,N) (col + row*N)
#endif

#define FMAD_OP(x, y, z) (x*y + z)
#define FMAD_FMA(x, y, z) fma(x, y, z)
#define FMAD_MAD(x, y, z) mad(x, y, z)

#if MASK
#define CONDITIONAL_ACCUMULATE(condition, accumulator, a, b) \
  accumulator = FMAD(a, b*(condition), accumulator)
#else
#define CONDITIONAL_ACCUMULATE(condition, accumulator, a, b) \
  if (condition) accumulator = FMAD(a, b, accumulator)
#endif

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
    CONDITIONAL_ACCUMULATE(row != col, tmp, A[INDEX(col,row,N)], xold[col]);
  }
  xnew[row] = (b[row] - tmp) / A[INDEX(row,row,N)];
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
    CONDITIONAL_ACCUMULATE(row != col, tmp, A[INDEX(col,row,N)], xold[col]);
  }

  scratch[lid] = tmp;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (unsigned offset = lsz/2; offset > 0; offset/=2)
  {
    if (lid < offset)
      scratch[lid] += scratch[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  xnew[row] = (b[row] - scratch[0]) / A[INDEX(row,row,N)];
}
