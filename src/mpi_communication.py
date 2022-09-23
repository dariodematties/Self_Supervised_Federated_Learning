import numpy as np
import torch


def gather_tensor(comm, data, root=0):
    size = comm.Get_size()
    rank = comm.Get_rank()
    gpu = rank
    device = 'cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu'

    sendbuf = np.array(data.cpu())
    recvbuf = None
    shape=sendbuf.shape
    sendbuf = sendbuf.reshape([np.prod(shape)])
    if rank == root:
        recvbuf = np.empty([size, np.prod(shape)], dtype=sendbuf.dtype)

    comm.Gather(sendbuf, recvbuf, root=root)
    output=[]
    if rank == root:
        for array in recvbuf:
            output.append(torch.tensor(array.reshape(shape)).to(device))

    return output


def gather_state_dict(comm, state_dict, root=0):
    size = comm.Get_size()
    rank = comm.Get_rank()

    outputs=[]
    if rank==root:
        for i in range(size):
            outputs.append({})

    for param_tensor in state_dict:
        tensors=gather_tensor(comm, state_dict[param_tensor], root=root)

        if rank==root:
            for output, tensor in zip(outputs, tensors):
                output[param_tensor]=tensor

    return outputs
        

def gather_weights(comm, state_dicts, args, root=0):
    rank = comm.Get_rank()

    length=max(comm.allgather(len(state_dicts)))

    while len(state_dicts) < length:
        state_dicts.append(state_dicts[-1])

    outputs=[]
    for state_dict in state_dicts:
        auxiliary_state_dicts = gather_state_dict(comm, state_dict, root=root)

        if rank==root:
            outputs.extend(auxiliary_state_dicts)

    m = max(int(args.frac * args.num_users), 1)
    outputs=outputs[:m]

    return outputs












def bcast_state_dict(comm, state_dict, keys, root=0):
    rank = comm.Get_rank()
    output={}
    for param_tensor in keys:
        if rank==root:
            buf = state_dict[param_tensor]
            shape = buf.shape
            dtype = buf.dtype
        else:
            shape = None
            dtype = None

        shape = comm.bcast(shape, root=root)
        dtype = comm.bcast(dtype, root=root)

        if rank!=root:
            buf = torch.empty(shape, dtype=dtype)

        output[param_tensor] = bcast_tensor(comm, buf, root=root)

    return output


def bcast_tensor(comm, data, root=0):
    rank = comm.Get_rank()
    gpu = rank
    device = 'cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu'

    buf = np.array(data.cpu())
    shape=buf.shape

    buf = buf.reshape([np.prod(shape)])
    comm.Bcast(buf, root=root)

    output = torch.tensor(buf.reshape(shape)).to(device)

    return output





















def gather_losses(comm, losses, args, root=0):
    rank = comm.Get_rank()

    length=max(comm.allgather(len(losses)))

    while len(losses) < length:
        losses.append(losses[-1])

    outputs=[]
    for loss in losses:
        auxiliary_losses = comm.gather(loss, root=root)

        if rank==root:
            outputs.extend(auxiliary_losses)

    m = max(int(args.frac * args.num_users), 1)
    outputs=outputs[:m]

    return outputs

def gather_accuracies(comm, accs, args, root=0):
    rank = comm.Get_rank()

    length=max(comm.allgather(len(accs)))

    while len(accs) < length:
        accs.append(accs[-1])

    outputs=[]
    for acc in accs:
        auxiliary_losses = comm.gather(acc, root=root)

        if rank==root:
            outputs.extend(auxiliary_losses)

    m = max(int(args.frac * args.num_users), 1)
    outputs=outputs[:m]

    return outputs

