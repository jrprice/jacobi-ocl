import math
import numpy
import pyopencl as CL
import time

N       = 512
ITRS    = 2000
print 'N    = %d' % N
print 'ITRS = %d' % ITRS

# Initialize OpenCL objects
context = CL.create_some_context()
print 'Using \'' + context.devices[0].name + '\''
queue   = CL.CommandQueue(context)
program = CL.Program(context, open('kernel.cl').read()).build()
kernel  = program.jacobi

# Create buffers
d_A     = CL.Buffer(context, CL.mem_flags.READ_ONLY,  size=N*N*8)
d_b     = CL.Buffer(context, CL.mem_flags.READ_ONLY,  size=N*8)
d_x0    = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=N*8)
d_x1    = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=N*8)
d_xold  = d_x0
d_xnew  = d_x1

# Initialize data
numpy.random.seed(0)
h_A     = numpy.random.rand(N, N).astype(numpy.float64)
for row in range(N):
    h_A[row][row] += numpy.sum(h_A[row])
h_b     = numpy.random.rand(N).astype(numpy.float64)
h_x     = numpy.zeros(N).astype(numpy.float64)
CL.enqueue_copy(queue, d_A, h_A)
CL.enqueue_copy(queue, d_b, h_b)
CL.enqueue_copy(queue, d_xold, h_x)

kernel.set_arg(0, d_A)
kernel.set_arg(1, d_b)
kernel.set_arg(4, numpy.uint32(N))

# Run Jacobi iterations
start = time.time()
for i in range(ITRS):
    kernel.set_arg(2, d_xold)
    kernel.set_arg(3, d_xnew)
    CL.enqueue_nd_range_kernel(queue, kernel, (N,), None)

    d_xold,d_xnew = d_xnew,d_xold

CL.enqueue_copy(queue, h_x, d_xold)
queue.finish()
end = time.time()

print 'Runtime = %.3fs' % (end-start)
print 'Error = %f' % math.sqrt(sum([e*e for e in (h_b - numpy.dot(h_A, h_x))]))
