import h5py
import numpy as np

with h5py.File("python-script/data/glove-100-angular.hdf5", "r") as f:
    print(f["train"].dtype)
    print(f["neighbors"].dtype)
    print(f["test"].dtype)
    print(f["distances"].dtype)
    print(f["train"].shape)
    print(f["neighbors"].shape)
    print(f["test"].shape)
    print(f["distances"].shape)

    train = f["train"][:].astype("<f4", copy=False)
    n, dim = train.shape
    with open("python-script/data/train.bin", "wb") as out:
        out.write(np.array([n, dim], dtype="<i4").tobytes())
        out.write(np.ascontiguousarray(train).tobytes())

    test = f["test"][:].astype("<f4", copy=False)
    n, dim = test.shape
    with open("python-script/data/test.bin", "wb") as out:
        out.write(np.array([n, dim], dtype="<i4").tobytes())
        out.write(np.ascontiguousarray(test).tobytes())

    distances = f["distances"][:].astype("<f4", copy=False)
    n, dim = distances.shape
    with open("python-script/data/distances.bin", "wb") as out:
        out.write(np.array([n, dim], dtype="<i4").tobytes())
        out.write(np.ascontiguousarray(distances).tobytes())

    neighbors = f["neighbors"][:].astype("<i4", copy=False)
    n, dim = neighbors.shape
    with open("python-script/data/neighbors.bin", "wb") as out:
        out.write(np.array([n, dim], dtype="<i4").tobytes())
        out.write(np.ascontiguousarray(neighbors).tobytes())