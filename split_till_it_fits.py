import torch


class SplitSize:
    def __init__(self, name):
        assert isinstance(name, str)
        self._num_splits = 1
        self._name = name

    def get(self):
        return self._num_splits

    def name(self):
        return self._name

    def double(self):
        self._num_splits *= 2


def split_till_it_fits(fn, split_size, *args, **kwargs):
    assert isinstance(split_size, SplitSize)
    split_size_was_increased = False
    max_size = 1024 * 1024
    recoverable_exceptions = [
        "out of memory",
        "not enough memory",
        "This error may appear if you passed in a non-contiguous input",
        "CUBLAS_STATUS_ALLOC_FAILED",
    ]
    while split_size.get() <= max_size:
        try:
            ret = fn(*args, **kwargs, num_splits=split_size.get())
            if split_size_was_increased:
                print(
                    f'The split size for "{split_size.name()}" was increased to {split_size.get()}'
                )
            return ret
        except RuntimeError as e:
            se = str(e)
            oom = any([excp in se for excp in recoverable_exceptions])
            if not oom:
                raise e
            torch.cuda.empty_cache()
            split_size.double()
            split_size_was_increased = True
    raise Exception(
        f'The split size for "{split_size.name()}" was increased too much and there\'s probably a bug'
    )
