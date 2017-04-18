package lazyseq

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"fmt"
	"io"
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
func CompressedTape(level int) (Tape, chan<- *anyseq.Batch) {
	var creatorLock sync.RWMutex
	var creator anyvec.Creator

	return newAbstractTape(func(in interface{}) *anyseq.Batch {
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
		return compressBatch(level, b)
	})
}

type compressedBatch struct {
	Present []bool
	Data    []byte
	Float32 bool
}

func compressBatch(level int, b *anyseq.Batch) *compressedBatch {
	var binaryData bytes.Buffer
	vecData := b.Packed.Data()
	_, is32Bit := vecData.([]float32)
	if !is32Bit {
		if _, ok := vecData.([]float64); !ok {
			panic(fmt.Sprintf("CompressedTape: unsupported anyvec.NumericList: %T",
				vecData))
		}
	}
	binary.Write(&binaryData, binary.LittleEndian, vecData)

	var compressedData bytes.Buffer
	w, err := flate.NewWriter(&compressedData, level)

	// Only throws an error if the level is invalid.
	if err != nil {
		panic(err)
	}

	io.Copy(w, &binaryData)
	w.Close()

	return &compressedBatch{
		Present: b.Present,
		Data:    compressedData.Bytes(),
		Float32: is32Bit,
	}
}

func (c *compressedBatch) Decompress(cr anyvec.Creator) *anyseq.Batch {
	compressed := bytes.NewReader(c.Data)
	var origData bytes.Buffer
	reader := flate.NewReader(compressed)
	if _, err := io.Copy(&origData, reader); err != nil {
		panic(err)
	}

	var vec anyvec.NumericList
	if c.Float32 {
		vec = make([]float32, origData.Len()/4)
	} else {
		vec = make([]float64, origData.Len()/8)
	}
	if err := binary.Read(&origData, binary.LittleEndian, vec); err != nil {
		panic(err)
	}

	return &anyseq.Batch{
		Present: c.Present,
		Packed:  cr.MakeVectorData(vec),
	}
}
