import argparse
import json
import math
import numpy
import pyopencl as CL
import time

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--norder', type=int, default=256)
parser.add_argument('-i', '--iterations', type=int, default=1000)
parser.add_argument('-c', '--config', default='')
args = parser.parse_args()

if args.config:
    # Load config from JSON file
    with open(args.config) as config_file:
        config = json.load(config_file)
else:
    # Default configuration
    config = dict()
    config['wgsize'] = 64

SEPARATOR = '--------------------------------'
print SEPARATOR
print 'MATRIX     = %dx%d ' % (args.norder,args.norder)
print 'ITERATIONS = %d' % args.iterations
print SEPARATOR
print 'Work-group size = ' + str(config['wgsize'])
print SEPARATOR

# Validate configuration
if args.norder % config['wgsize']:
    print 'Invalid wgsize value (must divide matrix order)'
    exit(1)

# Initialize OpenCL objects
context = CL.create_some_context()
print 'Using \'' + context.devices[0].name + '\''
queue   = CL.CommandQueue(context)
program = CL.Program(context, open('kernel.cl').read()).build()
kernel  = program.jacobi

# Create buffers
vectorsize = args.norder*numpy.dtype(numpy.float64).itemsize
matrixsize = args.norder*vectorsize
d_A     = CL.Buffer(context, CL.mem_flags.READ_ONLY,  size=matrixsize)
d_b     = CL.Buffer(context, CL.mem_flags.READ_ONLY,  size=vectorsize)
d_x0    = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
d_x1    = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
d_xold  = d_x0
d_xnew  = d_x1

# Initialize data
numpy.random.seed(0)
h_A     = numpy.random.rand(args.norder, args.norder).astype(numpy.float64)
for row in range(args.norder):
    h_A[row][row] += numpy.sum(h_A[row])
h_b     = numpy.random.rand(args.norder).astype(numpy.float64)
h_x     = numpy.zeros(args.norder).astype(numpy.float64)
CL.enqueue_copy(queue, d_A, h_A)
CL.enqueue_copy(queue, d_b, h_b)
CL.enqueue_copy(queue, d_xold, h_x)

local_size  = (config['wgsize'],)
global_size = (args.norder,)

kernel.set_arg(0, d_A)
kernel.set_arg(1, d_b)
kernel.set_arg(4, numpy.uint32(args.norder))

# Run Jacobi iterations
start = time.time()
for i in range(args.iterations):
    kernel.set_arg(2, d_xold)
    kernel.set_arg(3, d_xnew)
    CL.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)

    d_xold,d_xnew = d_xnew,d_xold

CL.enqueue_copy(queue, h_x, d_xold)
queue.finish()
end = time.time()

print 'Runtime = %.3fs' % (end-start)
print 'Error = %f' % math.sqrt(sum([e*e for e in (h_b - numpy.dot(h_A, h_x))]))
