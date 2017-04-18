package test

import (
	"compress/flate"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestCompressedTape(t *testing.T) {
	t.Run("Float32", func(t *testing.T) {
		tape, writer := lazyseq.CompressedTape(flate.DefaultCompression)
		testTapeOps(t, anyvec32.DefaultCreator{}, tape, writer)
	})
	t.Run("Float64", func(t *testing.T) {
		tape, writer := lazyseq.CompressedTape(flate.DefaultCompression)
		testTapeOps(t, anyvec64.DefaultCreator{}, tape, writer)
	})
}

func BenchmarkCompressedTape(b *testing.B) {
	c := anyvec32.DefaultCreator{}

	// Simulate compressing 16 frames from Atari Pong.
	batch := &anyseq.Batch{
		Present: make([]bool, 16),
		Packed:  c.MakeVector(160 * 210 * 16),
	}
	for i := range batch.Present {
		batch.Present[i] = true
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tape, writer := lazyseq.CompressedTape(flate.DefaultCompression)
		r := tape.ReadTape(0, 1)
		writer <- batch
		<-r
		close(writer)
	}
}
