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
parser.add_argument('-k', '--convergence-frequency', type=int, default=0)
parser.add_argument('-t', '--convergence-tolerance', type=float, default=0.001)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

# Default configuration
config = dict()
config['wgsize']       = 64
config['kernel']       = 'row_per_wi'
config['conditional']  = 'branch'
config['fmad']         = 'op'
config['addrspace_b']  = 'global'
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
if args.convergence_frequency:
    print 'Check convergence every %d iterations (tolerance=%g)' \
        % (args.convergence_frequency, args.convergence_tolerance)
else:
    print 'Convergence checking disabled'
print SEPARATOR
print 'Work-group size = ' + str(config['wgsize'])
print 'Kernel type     = ' + config['kernel']
print 'Conditional     = ' + config['conditional']
print 'fmad            = ' + config['fmad']
print 'b address space = ' + config['addrspace_b']
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
if not config['addrspace_b'] in ['global', 'constant']:
    print 'Invalid value for addrspace_b (must be \'global\' or \'constant\')'
    exit(1)
build_options += ' -DADDRSPACE_B=' + str(config['addrspace_b'])
if config['conditional'] == 'mask':
    build_options += ' -DMASK=1'
elif config['conditional'] != 'branch':
    print 'Invalid conditional value (must be \'branch\' or \'predicate\')'
    exit(1)
if config['fmad'] == 'op':
    build_options += ' -DFMAD=FMAD_OP'
elif config['fmad'] == 'fma':
    build_options += ' -DFMAD=FMAD_FMA'
elif config['fmad'] == 'mad':
    build_options += ' -DFMAD=FMAD_MAD'
else:
    print 'Invalid fmad value (must be \'op\' or \'fma\' or \'mad\')'
    exit(1)

if args.verbose:
    print
    print 'Build options:'
    print build_options
    print

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
    jacobi      = program.jacobi_row_per_wi
elif config['kernel'] == 'row_per_wg':
    global_size = (args.norder*local_size[0],)
    jacobi      = program.jacobi_row_per_wg
    jacobi.set_arg(5, CL.LocalMemory(local_size[0]*typesize))
else:
    print 'Invalid kernel type'
    exit(1)

jacobi.set_arg(0, numpy.uint32(args.norder))
jacobi.set_arg(3, d_A)
jacobi.set_arg(4, d_b)

num_groups  = global_size[0] / local_size[0]
h_err       = numpy.zeros(num_groups)
d_err       = CL.Buffer(context, CL.mem_flags.WRITE_ONLY,
                        size=num_groups*typesize)
convergence = program.convergence
convergence.set_arg(0, numpy.uint32(args.norder))
convergence.set_arg(1, d_x0)
convergence.set_arg(2, d_x1)
convergence.set_arg(3, d_err)
convergence.set_arg(4, CL.LocalMemory(local_size[0]*typesize))

# Run Jacobi iterations
start = time.time()
for i in range(args.iterations):
    jacobi.set_arg(1, d_xold)
    jacobi.set_arg(2, d_xnew)
    CL.enqueue_nd_range_kernel(queue, jacobi, global_size, local_size)

    # Convergence check
    if args.convergence_frequency and (i+1)%args.convergence_frequency == 0:
        CL.enqueue_nd_range_kernel(queue, convergence,
                                   (args.norder,), local_size)
        CL.enqueue_copy(queue, h_err, d_err)
        queue.finish()
        if math.sqrt(numpy.sum(h_err)) < args.convergence_tolerance:
            break

    d_xold,d_xnew = d_xnew,d_xold

CL.enqueue_copy(queue, h_x, d_xold)
queue.finish()
end = time.time()

print 'Runtime = %.3fs (%d iterations)' % ((end-start), i+1)
print 'Error   = %f' % math.sqrt(sum([e*e for e in (h_b-numpy.dot(h_A, h_x))]))
