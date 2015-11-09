import argparse
import json
import math
import numpy
import pyopencl as CL
import time

def run(config, norder, iterations,
        convergence_frequency=0, convergence_tolerance=0.001):

    # Print configuration
    SEPARATOR = '--------------------------------'
    print SEPARATOR
    print 'MATRIX     = %dx%d ' % (norder,norder)
    print 'ITERATIONS = %d' % iterations
    if convergence_frequency:
        print 'Check convergence every %d iterations (tolerance=%g)' \
            % (convergence_frequency, convergence_tolerance)
    else:
        print 'Convergence checking disabled'
    print SEPARATOR
    print 'Work-group size = ' + str(config['wgsize'])
    print 'Unroll factor   = ' + str(config['unroll'])
    print 'Kernel type     = ' + config['kernel']
    print 'Data layout     = ' + config['layout']
    print 'Conditional     = ' + config['conditional']
    print 'fmad            = ' + config['fmad']
    print 'b address space = ' + config['addrspace_b']
    print 'Integer type    = ' + config['integer']
    print 'Relaxed math    = ' + str(config['relaxed_math'])
    print 'Use mad24       = ' + str(config['use_mad24'])
    print 'Constant norder = ' + str(config['const_norder'])
    print SEPARATOR

    # Ensure work-group size is valid
    if norder % config['wgsize']:
        print 'Invalid wgsize value (must divide matrix order)'
        exit(1)

    # Initialize OpenCL objects
    context = CL.create_some_context()
    queue   = CL.CommandQueue(context)
    print 'Using \'' + context.devices[0].name + '\''

    # Create and build program
    build_options  = ''
    build_options += ' -cl-fast-relaxed-math' if config['relaxed_math'] else ''
    if config['const_norder']:
        build_options += ' -Dnorder=' + str(norder)
        if config['integer'] == 'uint':
            build_options += 'u'
    kernel_source  = generate_kernel(config, norder)
    program        = CL.Program(context, kernel_source).build(build_options)

    # Create buffers
    typesize   = numpy.dtype(numpy.float64).itemsize
    vectorsize = norder*typesize
    matrixsize = norder*norder*typesize
    d_A     = CL.Buffer(context, CL.mem_flags.READ_ONLY,  size=matrixsize)
    d_b     = CL.Buffer(context, CL.mem_flags.READ_ONLY,  size=vectorsize)
    d_x0    = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
    d_x1    = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
    d_xold  = d_x0
    d_xnew  = d_x1

    # Initialize data
    numpy.random.seed(0)
    h_A     = numpy.random.rand(norder, norder).astype(numpy.float64)
    for row in range(norder):
        h_A[row][row] += numpy.sum(h_A[row])
    h_b     = numpy.random.rand(norder).astype(numpy.float64)
    h_x     = numpy.zeros(norder).astype(numpy.float64)
    CL.enqueue_copy(queue, d_A, h_A)
    CL.enqueue_copy(queue, d_b, h_b)
    CL.enqueue_copy(queue, d_xold, h_x)

    # Create kernel and set invariant arguments
    jacobi      = program.jacobi
    if not config['const_norder']:
        jacobi.set_arg(0, numpy.uint32(norder))
        arg_xold = 1
    else:
        arg_xold = 0
    arg_xnew    = arg_xold + 1
    arg_A       = arg_xnew + 1
    arg_b       = arg_A    + 1
    arg_scratch = arg_b    + 1

    # Compute global size
    local_size = (config['wgsize'],)
    if config['kernel'] == 'row_per_wi':
        global_size = (norder,)
    elif config['kernel'] == 'row_per_wg':
        global_size = (norder*local_size[0],)
        jacobi.set_arg(arg_scratch, CL.LocalMemory(local_size[0]*typesize))
    else:
        print 'Invalid kernel type'
        exit(1)

    num_groups  = norder / local_size[0]
    h_err       = numpy.zeros(num_groups)
    d_err       = CL.Buffer(context, CL.mem_flags.WRITE_ONLY,
                            size=num_groups*typesize)
    convergence = program.convergence
    convergence.set_arg(0, d_x0)
    convergence.set_arg(1, d_x1)
    convergence.set_arg(2, d_err)
    convergence.set_arg(3, CL.LocalMemory(local_size[0]*typesize))

    if config['layout'] == 'col-major':
        # Run kernel to transpose data on device
        d_A_colmaj = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=matrixsize)
        transpose  = program.transpose
        transpose.set_arg(0, d_A)
        transpose.set_arg(1, d_A_colmaj)
        CL.enqueue_nd_range_kernel(queue, transpose, (norder,norder), None)
        d_A = d_A_colmaj

    jacobi.set_arg(arg_A, d_A)
    jacobi.set_arg(arg_b, d_b)

    # Run Jacobi solver
    start = time.time()
    for i in range(iterations):
        jacobi.set_arg(arg_xold, d_xold)
        jacobi.set_arg(arg_xnew, d_xnew)
        CL.enqueue_nd_range_kernel(queue, jacobi, global_size, local_size)

        # Convergence check
        if convergence_frequency and (i+1)%convergence_frequency == 0:
            CL.enqueue_nd_range_kernel(queue, convergence,
                                       (norder,), local_size)
            CL.enqueue_copy(queue, h_err, d_err)
            queue.finish()
            if math.sqrt(numpy.sum(h_err)) < convergence_tolerance:
                break

        d_xold,d_xnew = d_xnew,d_xold

    queue.finish()
    end = time.time()

    # Read results
    CL.enqueue_copy(queue, h_x, d_xold)

    # Print runtime and final error
    runtime = (end-start)
    error   = math.sqrt(sum([e*e for e in (h_b - numpy.dot(h_A, h_x))]))
    print 'Runtime = %.3fs (%d iterations)' % (runtime, i+1)
    print 'Error   = %f' % error

def generate_kernel(config, norder):

    def gen_index(config, col, row, N):
        if config['layout'] == 'row-major':
            x,y = col,row
        elif config['layout'] == 'col-major':
            x,y = row,col
        else:
            raise ValueError('layout', 'must be \'row-major\' or \'col-major\'')

        if config['use_mad24']:
            return 'mad24(%s, %s, %s)' % (y, N, x)
        else:
            return '(%s*%s + %s)' % (y, N, x)

    def gen_fmad(config, x, y, z):
        if config['fmad'] == 'op':
            return '(%s * %s + %s)' % (x, y, z)
        elif config['fmad'] == 'fma':
            return 'fma(%s, %s, %s)' % (x, y, z)
        elif config['fmad'] == 'mad':
            return 'mad(%s, %s, %s)' % (x, y, z)
        else:
            raise ValueError('fmad', 'must be \'op\' or \'fma\' or \'mad\')')

    def gen_cond_accum(config, cond, acc, a, b):
        result = ''
        if config['conditional'] == 'branch':
            result += 'if (%s) ' % cond
            _b = b
        elif config['conditional'] == 'mask':
            _b = '%s*(%s)' % (b, cond)
        else:
            raise ValueError('conditional', 'must be \'branch\' or \'mask\'')
        result += '%s = %s' % (acc, gen_fmad(config, a, _b, acc))
        return result

    # Ensure addrspace_b value is valid
    if not config['addrspace_b'] in ['global', 'constant']:
        raise ValueError('addrspace_b', 'must be \'global\' or \'constant\'')

    # Ensure integer value is valid
    if not config['integer'] in ['uint', 'int']:
        raise ValueError('integer', 'must be \'uint\' or \'or\'')
    inttype = str(config['integer'])

    # Ensure unroll factor is valid
    cols_per_wi = norder
    if config['kernel'] == 'row_per_wg':
        cols_per_wi /= config['wgsize']
    if cols_per_wi % config['unroll']:
        print 'Invalid unroll factor (must exactly divide %d)' % cols_per_wi
        exit(1)

    result = ''

    result += 'kernel void jacobi('

    # Kernel arguments
    if not config['const_norder']:
        result += '\n  const  %s norder,' % inttype
    result += '\n  global double *xold,'
    result += '\n  global double *xnew,'
    result += '\n  global double *A,'
    result += '\n  ' + str(config['addrspace_b']) + ' double *b,'
    if config['kernel'] == 'row_per_wg':
        result += '\n  local  double *scratch,'
    result = result[:-1]
    result += ')'

    # Start of kernel
    result += '\n{'

    # Get row index
    if config['kernel'] == 'row_per_wi':
        result += '\n  %s row = get_global_id(0);' % inttype
        col_start = '0'
        col_inc   = '1'
    elif config['kernel'] == 'row_per_wg':
        result += '\n  %s row = get_group_id(0);' % inttype
        result += '\n  %s lid = get_local_id(0);' % inttype
        result += '\n  %s lsz = get_local_size(0);' % inttype
        col_start = 'lid'
        col_inc   = 'lsz'
    else:
        raise ValueError('kernel', 'must be \'row_per_wi\' or \'row_per_wg\'')

    # Initialise accumulator
    result += '\n\n  double tmp = 0.0;'

    # Loop begin
    result += '\n  for (%s col = %s; col < norder; )' % (inttype, col_start)
    result += '\n  {'

    # Loop body
    A          = 'A[' + gen_index(config,'col','row','norder') + ']'
    x          = 'xold[col]'
    loop_body  = '\n    %s;' % gen_cond_accum(config, 'row != col', 'tmp', A, x)
    loop_body += '\n    col += %s;' % col_inc
    result += loop_body * config['unroll']

    # Loop end
    result += '\n  }\n'

    # xnew = (b - tmp) / D
    D = 'A[' + gen_index(config,'row','row','norder') + ']'
    if config['kernel'] == 'row_per_wi':
        result += '\n  xnew[row] = (b[row] - tmp) / %s;' % D
    elif config['kernel'] == 'row_per_wg':
        result += '\n  scratch[lid] = tmp;'
        result += '\n  barrier(CLK_LOCAL_MEM_FENCE);'
        result += '\n  for (%s offset = lsz/2; offset > 0; offset/=2)' % inttype
        result += '\n  {'
        result += '\n    if (lid < offset)'
        result += '\n      scratch[lid] += scratch[lid + offset];'
        result += '\n    barrier(CLK_LOCAL_MEM_FENCE);'
        result += '\n  }'
        result += '\n  if (lid == 0)'
        result += '\n    xnew[row] = (b[row] - scratch[0]) / %s;' % D

    # End of kernel
    result += '\n}\n'

    # Convergence checking kernel
    result += '''
kernel void transpose(global double *input, global double *output)
{
  int row = get_global_id(0);
  int col = get_global_id(1);
  int n   = get_global_size(0);
  output[row*n + col] = input[col*n + row];
}

kernel void convergence(global double *x0,
                        global double *x1,
                        global double *result,
                        local  double *scratch)
{
  uint row = get_global_id(0);
  uint lid = get_local_id(0);
  uint lsz = get_local_size(0);

  double diff = x0[row] - x1[row];
  scratch[lid] = diff*diff;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = lsz/2; offset > 0; offset/=2)
  {
    if (lid < offset)
      scratch[lid] += scratch[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    result[get_group_id(0)] = scratch[0];
}
    '''

    return result

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--norder',
                        type=int, default=256)
    parser.add_argument('-i', '--iterations',
                        type=int, default=1000)
    parser.add_argument('-c', '--config',
                        default='')
    parser.add_argument('-k', '--convergence-frequency',
                        type=int, default=0)
    parser.add_argument('-t', '--convergence-tolerance',
                        type=float, default=0.001)
    parser.add_argument('-p', '--print-kernel',
                        action='store_true')
    args = parser.parse_args()

    # Default configuration
    config = dict()
    config['wgsize']       = 64
    config['unroll']       = 1
    config['kernel']       = 'row_per_wi'
    config['layout']       = 'row-major'
    config['conditional']  = 'branch'
    config['fmad']         = 'op'
    config['addrspace_b']  = 'global'
    config['integer']      = 'uint'
    config['relaxed_math'] = False
    config['use_mad24']    = False
    config['const_norder'] = False

    # Load config from JSON file
    if args.config:
        with open(args.config) as config_file:
            config.update(json.load(config_file))

    if args.print_kernel:
        print generate_kernel(config, args.norder)
        exit(0)

    # Run Jacobi solver
    run(config, args.norder, args.iterations,
        args.convergence_frequency, args.convergence_tolerance)

if __name__ == '__main__':
    main()
