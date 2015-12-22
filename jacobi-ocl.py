import argparse
import json
import math
import numpy
import pyopencl as CL
import time

def run(config, norder, iterations, device,
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
    print 'Work-group size    = ' + str(config['wgsize'])
    print 'Unroll factor      = ' + str(config['unroll'])
    print 'Data layout        = ' + config['layout']
    print 'Conditional        = ' + config['conditional']
    print 'fmad               = ' + config['fmad']
    print 'Divide by A        = ' + config['divide_A']
    print 'b address space    = ' + config['addrspace_b']
    print 'xold address space = ' + config['addrspace_xold']
    print 'Integer type       = ' + config['integer']
    print 'Relaxed math       = ' + str(config['relaxed_math'])
    print 'Use restrict       = ' + str(config['use_restrict'])
    print 'Use const pointers = ' + str(config['use_const'])
    print 'Use mad24          = ' + str(config['use_mad24'])
    print 'Constant norder    = ' + str(config['const_norder'])
    print 'Constant wgsize    = ' + str(config['const_wgsize'])
    print 'Coalesce columns   = ' + str(config['coalesce_cols'])
    print SEPARATOR

    # Ensure work-group size is valid
    if config['wgsize'][0] & (config['wgsize'][0]-1):
        print 'Invalid wgsize[0] value (must be power of two)'
        exit(1)
    if norder % config['wgsize'][1]:
        print 'Invalid wgsize[1] value (must divide matrix order)'
        exit(1)

    # Initialize OpenCL objects
    if device:
        context = CL.Context([device])
    else:
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
    arg_index = 0
    jacobi    = program.jacobi
    if not config['const_norder']:
        jacobi.set_arg(arg_index, numpy.uint32(norder))
        arg_index += 1
    arg_xold    = arg_index
    arg_index  += 1
    arg_xnew    = arg_index
    arg_index  += 1
    arg_A       = arg_index
    arg_index  += 1
    arg_b       = arg_index
    arg_index  += 1

    # Compute global size
    local_size = (config['wgsize'][0],config['wgsize'][1])
    global_size = (local_size[0],norder)
    if config['wgsize'][0] > 1:
        jacobi.set_arg(arg_index,
                       CL.LocalMemory(local_size[0]*local_size[1]*typesize))
        arg_index += 1

    if config['layout'] == 'col-major':
        # Run kernel to transpose data on device
        d_A_colmaj = CL.Buffer(context, CL.mem_flags.READ_WRITE,
                               size=matrixsize)
        transpose  = program.transpose
        transpose.set_arg(0, d_A)
        transpose.set_arg(1, d_A_colmaj)
        CL.enqueue_nd_range_kernel(queue, transpose, (norder,norder), None)
        d_A = d_A_colmaj

    if config['divide_A'] in ['precompute-global','precompute-constant']:
        # Run kernel to precompute 1/A for diagonal
        d_inv_A = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
        precompute_inv_A = program.precompute_inv_A
        precompute_inv_A.set_arg(0, d_A)
        precompute_inv_A.set_arg(1, d_inv_A)
        CL.enqueue_nd_range_kernel(queue, precompute_inv_A, (norder,), None)
        jacobi.set_arg(arg_index, d_inv_A)
        arg_index += 1

    jacobi.set_arg(arg_A, d_A)
    jacobi.set_arg(arg_b, d_b)

    # Prepare convergence checking kernel
    conv_wgsize = 64 # TODO: Pick something else? (e.g wgsize[0]*wgsize[1])
    num_groups  = norder / conv_wgsize
    h_err       = numpy.zeros(num_groups)
    d_err       = CL.Buffer(context, CL.mem_flags.WRITE_ONLY,
                            size=num_groups*typesize)
    convergence = program.convergence
    convergence.set_arg(0, d_x0)
    convergence.set_arg(1, d_x1)
    convergence.set_arg(2, d_err)
    convergence.set_arg(3, CL.LocalMemory(conv_wgsize*typesize))

    # Run Jacobi solver
    start = time.time()
    for i in range(iterations):
        jacobi.set_arg(arg_xold, d_xold)
        jacobi.set_arg(arg_xnew, d_xnew)
        CL.enqueue_nd_range_kernel(queue, jacobi, global_size, local_size)

        # Convergence check
        if convergence_frequency and (i+1)%convergence_frequency == 0:
            CL.enqueue_nd_range_kernel(queue, convergence,
                                       (norder,), (conv_wgsize,))
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

    def gen_ptrarg(config, addrspace, name, readonly=True):
        const    = 'const' if readonly and config['use_const'] else ''
        restrict = 'restrict' if config['use_restrict'] else ''
        ptrarg   = '%-8s %-5s double *%s %s'
        return ptrarg % (addrspace, const, restrict, name)

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

    def gen_divide_A(config, numerator):
        index = gen_index(config,'row','row','norder')
        if config['divide_A'] == 'normal':
            return '(%s) / A[%s]' % (numerator, index)
        elif config['divide_A'] == 'native':
            return 'native_divide(%s, A[%s])' % (numerator, index)
        elif config['divide_A'] in ['precompute-global','precompute-constant']:
            return '(%s) * inv_A[row]' % numerator
        else:
            raise ValueError('divide_A',
                             'must be \'normal\' or \'native\' or '+
                             '\'precompute-global\' or \'precompute-constant\'')

    # Ensure addrspace_b value is valid
    if not config['addrspace_b'] in ['global', 'constant']:
        raise ValueError('addrspace_b', 'must be \'global\' or \'constant\'')

    # Ensure addrspace_xold value is valid
    if not config['addrspace_xold'] in ['global', 'constant']:
        raise ValueError('addrspace_xold', 'must be \'global\' or \'constant\'')

    # Ensure integer value is valid
    if not config['integer'] in ['uint', 'int']:
        raise ValueError('integer', 'must be \'uint\' or \'or\'')
    inttype = str(config['integer'])

    # Ensure unroll factor is valid
    cols_per_wi = norder / config['wgsize'][0]
    if cols_per_wi % config['unroll']:
        print 'Invalid unroll factor (must exactly divide %d)' % cols_per_wi
        exit(1)

    row  = 'get_global_id(1)'

    lidx = 'get_local_id(0)'
    lidy = 'get_local_id(1)'
    lszx = 'get_local_size(0)'
    lszy = 'get_local_size(1)'
    if config['const_wgsize']:
        if config['wgsize'][0] == 1:
            lidx = '0'
        if config['wgsize'][1] == 1:
            lidy = '0'
        lszx = config['wgsize'][0]
        lszy = config['wgsize'][1]

    result = ''

    # Enable FP64 extension for OpenCL 1.1 devices
    result += '\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n'

    result += '\n kernel void jacobi('

    # Kernel arguments
    if not config['const_norder']:
        result += '\n  const %s norder,' % inttype
    result += '\n  %s,' % gen_ptrarg(config, config['addrspace_xold'], 'xold')
    result += '\n  %s,' % gen_ptrarg(config, 'global', 'xnew', False)
    result += '\n  %s,' % gen_ptrarg(config, 'global', 'A')
    result += '\n  %s,' % gen_ptrarg(config, config['addrspace_b'], 'b')
    if config['wgsize'][0] > 1:
        result += '\n  %s,' % gen_ptrarg(config, 'local', 'scratch', False)
    if config['divide_A'] == 'precompute-global':
        result += '\n  %s,' % gen_ptrarg(config, 'global', 'inv_A')
    elif config['divide_A'] == 'precompute-constant':
        result += '\n  %s,' % gen_ptrarg(config, 'constant', 'inv_A')
    result = result[:-1]
    result += ')'

    # Start of kernel
    result += '\n{'

    # Get row index
    result += '\n  const %s row  = %s;' % (inttype,row)
    result += '\n  const %s lidx = %s;' % (inttype,lidx)
    result += '\n  const %s lszx = %s;' % (inttype,lszx)

    # Get column range for work-item
    if config['coalesce_cols']:
        col_beg = 'lidx'
        col_end = 'norder'
        col_inc = 'lszx'
    else:
        col_beg = 'lidx*%d' % cols_per_wi
        col_end = '%s+%d' % (col_beg,cols_per_wi)
        col_inc = '1'

    # Initialise accumulator
    result += '\n\n  double tmp = 0.0;'

    # Loop begin
    result += '\n  for (%s col = %s; col < %s; )' % (inttype, col_beg, col_end)
    result += '\n  {'

    # Loop body
    A          = 'A[%s]' % gen_index(config,'col','row','norder')
    x          = 'xold[col]'
    loop_body  = '\n    %s;' % gen_cond_accum(config, 'row != col', 'tmp', A, x)
    loop_body += '\n    col += %s;' % col_inc
    result    += loop_body * config['unroll']

    # Loop end
    result += '\n  }\n'

    # xnew = (b - tmp) / D
    if config['wgsize'][0] > 1:
        result += '\n  int lid = %s + %s*%s;' % (lidx,lidy,lszx)
        result += '\n  scratch[lid] = tmp;'
        result += '\n  barrier(CLK_LOCAL_MEM_FENCE);'
        result += '\n  for (%s offset = lszx/2; offset>0; offset/=2)' % inttype
        result += '\n  {'
        result += '\n    if (lidx < offset)'
        result += '\n      scratch[lid] += scratch[lid + offset];'
        result += '\n    barrier(CLK_LOCAL_MEM_FENCE);'
        result += '\n  }'
        result += '\n  if (lidx == 0)'
        xnew    = gen_divide_A(config, 'b[row] - scratch[lid]')
        result += '\n    xnew[row] = %s;' % xnew
    else:
        xnew    = gen_divide_A(config, 'b[row] - tmp')
        result += '\n  xnew[row] = %s;' % xnew

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

kernel void precompute_inv_A(global double *A, global double *inv_A)
{
  int row = get_global_id(0);
  int n   = get_global_size(0);
  inv_A[row] = 1 / A[row*n + row];
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

    return str(result)

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
    parser.add_argument('-l', '--list',
                        action='store_true')
    parser.add_argument('-d', '--device',
                        type=int, default=0)
    args = parser.parse_args()

    # Print device list
    if args.list:
        devices = get_device_list()
        if devices:
            print
            print 'OpenCL devices:'
            for i in range(len(devices)):
                print '  %d: %s' % (i,devices[i].name)
            print
        else:
            print 'No OpenCL devices found'
        exit(0)

    # Default configuration
    config = dict()
    config['wgsize']         = [64,1]
    config['unroll']         = 1
    config['layout']         = 'row-major'
    config['conditional']    = 'branch'
    config['fmad']           = 'op'
    config['divide_A']       = 'normal'
    config['addrspace_b']    = 'global'
    config['addrspace_xold'] = 'global'
    config['integer']        = 'uint'
    config['relaxed_math']   = False
    config['use_const']      = False
    config['use_restrict']   = False
    config['use_mad24']      = False
    config['const_norder']   = False
    config['const_wgsize']   = False
    config['coalesce_cols']  = True

    # Load config from JSON file
    if args.config:
        with open(args.config) as config_file:
            config.update(json.load(config_file))

    if args.print_kernel:
        print generate_kernel(config, args.norder)
        exit(0)

    # Run Jacobi solver
    run(config, args.norder, args.iterations, get_device_list()[args.device],
        args.convergence_frequency, args.convergence_tolerance)

def get_device_list():
    platforms = CL.get_platforms()
    devices   = []
    for p in platforms:
        devices += p.get_devices()
    return devices

if __name__ == '__main__':
    main()
