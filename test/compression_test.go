package test

import (
	"compress/flate"
	"math/rand"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestCompressedTape(t *testing.T) {
	t.Run("Float32", func(t *testing.T) {
		tape, writer := lazyseq.CompressedTape(anyvec32.DefaultCreator{},
			flate.DefaultCompression)
		testTapeOps(t, tape, writer, nil)
	})
	t.Run("Float64", func(t *testing.T) {
		tape, writer := lazyseq.CompressedTape(anyvec64.DefaultCreator{},
			flate.DefaultCompression)
		testTapeOps(t, tape, writer, nil)
	})
}

func TestCompressedUint8Tape(t *testing.T) {
	randGen := func(v anyvec.Vector) {
		nums := make([]float64, v.Len())
		for i := range nums {
			nums[i] = float64(rand.Intn(0x100))
		}
		v.SetData(v.Creator().MakeNumericList(nums))
	}
	t.Run("Float32", func(t *testing.T) {
		tape, writer := lazyseq.CompressedUint8Tape(anyvec32.DefaultCreator{},
			flate.DefaultCompression)
		testTapeOps(t, tape, writer, randGen)
	})
	t.Run("Float64", func(t *testing.T) {
		tape, writer := lazyseq.CompressedUint8Tape(anyvec64.DefaultCreator{},
			flate.DefaultCompression)
		testTapeOps(t, tape, writer, randGen)
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
		tape, writer := lazyseq.CompressedTape(c, flate.DefaultCompression)
		r := tape.ReadTape(0, 1)
		writer <- batch
		<-r
		close(writer)
	}
}

func BenchmarkCompressedTapeRead(b *testing.B) {
	c := anyvec32.DefaultCreator{}

	// Simulate compressing 16 frames from Atari Pong.
	batch := &anyseq.Batch{
		Present: make([]bool, 16),
		Packed:  c.MakeVector(160 * 210 * 16),
	}
	for i := range batch.Present {
		batch.Present[i] = true
	}

	tape, writer := lazyseq.CompressedTape(c, flate.DefaultCompression)
	writer <- batch
	close(writer)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		r := tape.ReadTape(0, 1)
		<-r
	}
}

func BenchmarkCompressedUint8Tape(b *testing.B) {
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
		tape, writer := lazyseq.CompressedUint8Tape(c, flate.DefaultCompression)
		r := tape.ReadTape(0, 1)
		writer <- batch
		<-r
		close(writer)
	}
}

func BenchmarkCompressedUint8TapeRead(b *testing.B) {
	c := anyvec32.DefaultCreator{}

	// Simulate compressing 16 frames from Atari Pong.
	batch := &anyseq.Batch{
		Present: make([]bool, 16),
		Packed:  c.MakeVector(160 * 210 * 16),
	}
	for i := range batch.Present {
		batch.Present[i] = true
	}

	tape, writer := lazyseq.CompressedUint8Tape(c, flate.DefaultCompression)
	writer <- batch
	close(writer)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		r := tape.ReadTape(0, 1)
		<-r
	}
}
