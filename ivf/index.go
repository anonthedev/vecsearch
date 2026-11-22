package ivf

import (
	"encoding/binary"
	"os"
)

func Save(path string, centers []float32, lists [][]int, dim int) {
	file, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	k := int32(len(lists))
	d := int32(dim)
	if err := binary.Write(file, binary.LittleEndian, &k); err != nil {
		panic(err)
	}
	if err := binary.Write(file, binary.LittleEndian, &d); err != nil {
		panic(err)
	}
	if err := binary.Write(file, binary.LittleEndian, centers); err != nil {
		panic(err)
	}

	for _, list := range lists {
		n := int32(len(list))
		if err := binary.Write(file, binary.LittleEndian, &n); err != nil {
			panic(err)
		}
		ids := make([]int32, len(list))
		for i, id := range list {
			ids[i] = int32(id)
		}
		if err := binary.Write(file, binary.LittleEndian, ids); err != nil {
			panic(err)
		}
	}
}

func Load(path string) ([]float32, [][]int) {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var k, dim int32
	if err := binary.Read(file, binary.LittleEndian, &k); err != nil {
		panic(err)
	}
	if err := binary.Read(file, binary.LittleEndian, &dim); err != nil {
		panic(err)
	}

	centers := make([]float32, int(k)*int(dim))
	if err := binary.Read(file, binary.LittleEndian, centers); err != nil {
		panic(err)
	}

	lists := make([][]int, k)
	for c := 0; c < int(k); c++ {
		var n int32
		if err := binary.Read(file, binary.LittleEndian, &n); err != nil {
			panic(err)
		}
		ids := make([]int32, n)
		if err := binary.Read(file, binary.LittleEndian, ids); err != nil {
			panic(err)
		}
		lists[c] = make([]int, n)
		for i, id := range ids {
			lists[c][i] = int(id)
		}
	}
	return centers, lists
}
