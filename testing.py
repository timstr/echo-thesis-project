from argparse import ArgumentParser

def make_receiver_indices(receivers, arrangement):
    assert(isinstance(receivers, int))
    assert(arrangement == "flat" or arrangement == "grid")

    assert(receivers >= 4 or arrangement == "flat")
    assert(receivers <= 16 or arrangement == "grid")

    if arrangement == "grid" and receivers >= 4:
        if receivers >= 16:
            numrows = 4
        else:
            numrows = 2
    else:
        numrows = 1

    numcols = receivers // numrows

    receiver_indices = []
    for i in range(numcols):
        if numcols == 1:
            idx_base = 32
        elif numcols == 2:
            idx_base = 16 + 32 * i
        else:
            idx_base = ((i * 64) // numcols) + ((16 // (numcols * 2)) & 0xFFF8)
        for j in range(numrows):
            receiver_indices.append(idx_base + (j * 3 if numrows == 2 else j))
    
    assert(len(receiver_indices) == receivers)

    for i in receiver_indices:
        print(i)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--receivers", type=int, choices=[1, 2, 4, 8, 16, 32, 64], dest="receivers", required=True)
    parser.add_argument("--arrangement", type=str, choices=["flat", "grid"], dest="arrangement")
    args = parser.parse_args()
    make_receiver_indices(args.receivers, args.arrangement)