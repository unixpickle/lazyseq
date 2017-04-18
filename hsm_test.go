package lazyrnn

import (
	"fmt"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestFixedHSMEquiv(t *testing.T) {
	const inSize = 3
	const outSize = 2

	c := anyvec64.DefaultCreator{}

	block := anyrnn.NewLSTM(c, inSize, outSize)

	for interval := 1; interval < 10; interval++ {
		for _, lazy := range []bool{false, true} {
			t.Run(fmt.Sprintf("Interval%d:%v", interval, lazy), func(t *testing.T) {
				inSeqs := testSeqs(c, inSize)
				actualFunc := func() anyseq.Seq {
					return Unlazify(FixedHSM(interval, lazy, Lazify(inSeqs), block))
				}
				expectedFunc := func() anyseq.Seq {
					return anyrnn.Map(inSeqs, block)
				}
				testEquivalent(t, actualFunc, expectedFunc)
			})
		}
	}
}

func TestRecursiveHSMEquiv(t *testing.T) {
	const inSize = 3
	const outSize = 2

	c := anyvec64.DefaultCreator{}

	block := anyrnn.NewLSTM(c, inSize, outSize)

	for interval := 1; interval < 10; interval++ {
		for partition := 2; partition < 10; partition++ {
			for _, lazy := range []bool{false, true} {
				name := fmt.Sprintf("%d:%d:%v", interval, partition, lazy)
				t.Run(name, func(t *testing.T) {
					inSeqs := testSeqs(c, inSize)
					actualFunc := func() anyseq.Seq {
						return Unlazify(RecursiveHSM(interval, partition,
							lazy, Lazify(inSeqs), block))
					}
					expectedFunc := func() anyseq.Seq {
						return anyrnn.Map(inSeqs, block)
					}
					testEquivalent(t, actualFunc, expectedFunc)
				})
			}
		}
	}
}

func BenchmarkBPTT(b *testing.B) {
	b.Run("Regular", func(b *testing.B) {
		c := anyvec32.DefaultCreator{}
		ins, ups, block := setupHSMBenchmark(c)

		inSeq := Unlazify(TapeRereader(c, ins))
		upstreamBatches := Unlazify(TapeRereader(c, ups)).Output()
		grad := anydiff.NewGrad(anynet.AllParameters(block)...)

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			out := anyrnn.Map(inSeq, block)
			out.Propagate(upstreamBatches, grad)
		}
	})
	b.Run("Streaming", func(b *testing.B) {
		benchmarkLazy(b, func(r Rereader, b anyrnn.Block) Seq {
			return Lazify(anyrnn.Map(Unlazify(r), b))
		})
	})
}

func BenchmarkFixedSqrtT(b *testing.B) {
	b.Run("Regular", func(b *testing.B) {
		benchmarkLazy(b, func(r Rereader, b anyrnn.Block) Seq {
			return FixedHSM(16, false, r, b)
		})
	})
	b.Run("Lazy", func(b *testing.B) {
		benchmarkLazy(b, func(r Rereader, b anyrnn.Block) Seq {
			return FixedHSM(16, true, r, b)
		})
	})
}

func BenchmarkRecursiveHSM(b *testing.B) {
	b.Run("Regular", func(b *testing.B) {
		benchmarkLazy(b, func(r Rereader, b anyrnn.Block) Seq {
			return RecursiveHSM(128, 2, false, r, b)
		})
	})
	b.Run("Lazy", func(b *testing.B) {
		benchmarkLazy(b, func(r Rereader, b anyrnn.Block) Seq {
			return RecursiveHSM(128, 2, true, r, b)
		})
	})
}

func benchmarkLazy(b *testing.B, f func(Rereader, anyrnn.Block) Seq) {
	c := anyvec32.DefaultCreator{}
	ins, ups, block := setupHSMBenchmark(c)
	grad := anydiff.NewGrad(anynet.AllParameters(block)...)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		out := f(TapeRereader(c, ins), block)
		upstream := make(chan *anyseq.Batch, 1)
		go func() {
			for b := range ups.ReadTape(0, -1) {
				upstream <- b
			}
			close(upstream)
		}()
		out.Propagate(upstream, NewGrad(grad))
	}
}

func setupHSMBenchmark(c anyvec.Creator) (inputs, upstream Tape, block anyrnn.Block) {
	const seqLen = 256
	const batchSize = 8
	const inSize = 64
	const outSize = 64

	block = anyrnn.NewLSTM(c, inSize, outSize)
	inputs, inputsCh := ReferenceTape()
	upstream, upstreamCh := ReferenceTape()

	for i := 0; i < seqLen; i++ {
		batch := &anyseq.Batch{
			Present: make([]bool, batchSize),
			Packed:  c.MakeVector(batchSize * inSize),
		}
		upBatch := &anyseq.Batch{
			Present: make([]bool, batchSize),
			Packed:  c.MakeVector(batchSize * inSize),
		}
		for i := range batch.Present {
			batch.Present[i] = true
			upBatch.Present[i] = true
		}
		inputsCh <- batch
		upstreamCh <- upBatch
	}

	close(inputsCh)
	close(upstreamCh)

	return
}
