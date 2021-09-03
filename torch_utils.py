import torch


def save_module(the_module, filename):
    print(f'Saving module to "{filename}"')
    torch.save(the_module.state_dict(), filename)


def restore_module(the_module, filename):
    print('Restoring module from "{}"'.format(filename))
    the_module.load_state_dict(torch.load(filename))
