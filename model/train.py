def LinearScheduler(optimizer, total_iter, curr, lr_init):
    lr = -(lr_init / total_iter) * curr + lr_init
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def batch_len(input_ids, pad_idx=0):
    b_length = (input_ids != pad_idx).data.sum(dim=-1)
    return b_length


