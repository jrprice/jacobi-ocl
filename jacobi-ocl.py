import argparse
import json
import math
import numpy
import pyopencl as CL
import time

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--norder', type=int, default=256)
parser.add_argument('-i', '--iterations', type=int, default=1000)
parser.add_argument('-c', '--config', default='')
args = parser.parse_args()

# Default configuration
config = dict()
config['wgsize']       = 64
config['kernel']       = 'row_per_wi'
config['conditional']  = 'branch'
config['relaxed_math'] = False
config['use_mad24']    = False

# Load config from JSON file
if args.config:
    with open(args.config) as config_file:
        config.update(json.load(config_file))

# Print configuration
SEPARATOR = '--------------------------------'
print SEPARATOR
print 'MATRIX     = %dx%d ' % (args.norder,args.norder)
print 'ITERATIONS = %d' % args.iterations
print SEPARATOR
print 'Work-group size = ' + str(config['wgsize'])
print 'Kernel type     = ' + config['kernel']
print 'Conditional     = ' + config['conditional']
print 'Relaxed math    = ' + str(config['relaxed_math'])
print 'Use mad24       = ' + str(config['use_mad24'])
print SEPARATOR

# Ensure work-group size is valid
if args.norder % config['wgsize']:
    print 'Invalid wgsize value (must divide matrix order)'
    exit(1)

# Initialize OpenCL objects
context = CL.create_some_context()
queue   = CL.CommandQueue(context)
print 'Using \'' + context.devices[0].name + '\''

# Create and build program
build_options  = ''
build_options += '-DUSE_MAD24=' + str(1 if config['use_mad24'] else 0)
build_options += ' -cl-fast-relaxed-math' if config['relaxed_math'] else ''
if config['conditional'] == 'predicate':
    build_options += ' -DPREDICATE=1'
elif config['conditional'] != 'branch':
    print 'Invalid conditional value (must be \'branch\' or \'predicate\')'
    exit(1)
print build_options
program = CL.Program(context, open('kernel.cl').read()).build(build_options)

# Create buffers
typesize   = numpy.dtype(numpy.float64).itemsize
vectorsize = args.norder*typesize
matrixsize = args.norder*args.norder*typesize
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

# Select kernel and global size
local_size = (config['wgsize'],)
if config['kernel'] == 'row_per_wi':
    global_size = (args.norder,)
    kernel = program.jacobi_row_per_wi
elif config['kernel'] == 'row_per_wg':
    global_size = (args.norder*local_size[0],)
    kernel = program.jacobi_row_per_wg
    kernel.set_arg(5, CL.LocalMemory(local_size[0]*typesize))
else:
    print 'Invalid kernel type'
    exit(1)

kernel.set_arg(0, numpy.uint32(args.norder))
kernel.set_arg(3, d_A)
kernel.set_arg(4, d_b)

# Run Jacobi iterations
start = time.time()
for i in range(args.iterations):
    kernel.set_arg(1, d_xold)
    kernel.set_arg(2, d_xnew)
    CL.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)

    d_xold,d_xnew = d_xnew,d_xold

CL.enqueue_copy(queue, h_x, d_xold)
queue.finish()
end = time.time()

print 'Runtime = %.3fs' % (end-start)
print 'Error = %f' % math.sqrt(sum([e*e for e in (h_b - numpy.dot(h_A, h_x))]))
