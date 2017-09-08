package lazyseq

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

// CompressedTape creates a Tape which stores vectors by
// converting them to an anyvec.NumericList, encoding that
// list as binary, and then compressing the binary data.
//
// The level argument is a compression level for the flate
// package.
// Typically, flate.DefaultCompression should be fine.
//
// The tape can be written via the returned channel.
//
// The anyvec.Creator should use []float32 or []float64 as
// its numeric type.
// Other types are not guaranteed to work.
//
// The caller must close the write channel to free
// resources associated with the Tape.
func CompressedTape(c anyvec.Creator, level int) (Tape, chan<- *anyseq.Batch) {
	return compressedTape(c, level, false)
}

// CompressedUint8Tape is like CompressedTape, but the
// tape only works for vectors whose components are whole
// numbers in the range [0, 255].
// This can yield better performance than CompressedTape
// when the data is known to be 8-bit unsigned integers
// beforehand.
func CompressedUint8Tape(c anyvec.Creator, level int) (Tape, chan<- *anyseq.Batch) {
	return compressedTape(c, level, true)
}

func compressedTape(c anyvec.Creator, level int, useUint8 bool) (Tape,
	chan<- *anyseq.Batch) {
	var creatorLock sync.RWMutex
	var creator anyvec.Creator

	return newAbstractTape(c, func(in interface{}) *anyseq.Batch {
		batch := in.(*compressedBatch)
		creatorLock.RLock()
		cr := creator
		creatorLock.RUnlock()
		return batch.Decompress(cr)
	}, func(b *anyseq.Batch) interface{} {
		creatorLock.Lock()
		if creator == nil {
			creator = b.Packed.Creator()
		} else if creator != b.Packed.Creator() {
			creatorLock.Unlock()
			panic("inconsistent anyvec.Creator in Tape")
		}
		creatorLock.Unlock()
		return compressBatch(level, useUint8, b)
	})
}

type compressedBatch struct {
	Present  []bool
	Data     []byte
	Float32  bool
	UseUint8 bool
}

func compressBatch(level int, useUint8 bool, b *anyseq.Batch) *compressedBatch {
	binaryData := &bytes.Buffer{}
	vecData := b.Packed.Data()
	list32, is32Bit := vecData.([]float32)
	if is32Bit {
		var encoded []byte
		if useUint8 {
			encoded = make([]byte, len(list32))
			for i, x := range list32 {
				encoded[i] = byte(x)
			}
		} else {
			encoded = make([]byte, len(list32)*4)
			for i, num := range list32 {
				data := math.Float32bits(num)
				idx := i << 2
				encoded[idx] = byte(data)
				encoded[idx+1] = byte(data >> 8)
				encoded[idx+2] = byte(data >> 16)
				encoded[idx+3] = byte(data >> 24)
			}
		}
		binaryData = bytes.NewBuffer(encoded)
	} else if list64, ok := vecData.([]float64); ok {
		if useUint8 {
			encoded := make([]byte, len(list64))
			for i, x := range list64 {
				encoded[i] = byte(x)
			}
			binaryData = bytes.NewBuffer(encoded)
		} else {
			binary.Write(binaryData, binary.LittleEndian, vecData)
		}
	} else {
		panic(fmt.Sprintf("CompressedTape: unsupported anyvec.NumericList: %T",
			vecData))
	}

	var compressedData bytes.Buffer
	w, err := flate.NewWriter(&compressedData, level)

	// Only throws an error if the level is invalid.
	if err != nil {
		panic(err)
	}

	io.Copy(w, binaryData)
	w.Close()

	return &compressedBatch{
		Present:  b.Present,
		Data:     compressedData.Bytes(),
		Float32:  is32Bit,
		UseUint8: useUint8,
	}
}

func (c *compressedBatch) Decompress(cr anyvec.Creator) *anyseq.Batch {
	compressed := bytes.NewReader(c.Data)
	var origData bytes.Buffer
	reader := flate.NewReader(compressed)
	if _, err := io.Copy(&origData, reader); err != nil {
		panic(err)
	}

	var numList anyvec.NumericList
	if c.Float32 {
		if c.UseUint8 {
			vec := make([]float32, origData.Len())
			for i, b := range origData.Bytes() {
				vec[i] = float32(b)
			}
			numList = vec
		} else {
			// This takes 25% as much time as a pure binary.Read()
			// on my machine.
			vec := make([]float32, origData.Len()/4)
			origBytes := origData.Bytes()
			for i := 0; i+4 <= len(origBytes); i += 4 {
				vec[i>>2] = math.Float32frombits(uint32(origBytes[i]) |
					(uint32(origBytes[i+1]) << 8) |
					(uint32(origBytes[i+2]) << 16) |
					(uint32(origBytes[i+3]) << 24))
			}
			numList = vec
		}
	} else {
		if c.UseUint8 {
			vec := make([]float64, origData.Len())
			for i, b := range origData.Bytes() {
				vec[i] = float64(b)
			}
			numList = vec
		} else {
			numList = make([]float64, origData.Len()/8)
			if err := binary.Read(&origData, binary.LittleEndian, numList); err != nil {
				panic(err)
			}
		}
	}

	return &anyseq.Batch{
		Present: c.Present,
		Packed:  cr.MakeVectorData(numList),
	}
}
