package test

import (
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestReduceTape(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	tape, writer := lazyseq.ReferenceTape(anyvec64.DefaultCreator{})

	writer <- &anyseq.Batch{
		Packed:  c.MakeVectorData([]float64{1, 2, 3, 4, 5, 6}),
		Present: []bool{true, false, true, true},
	}
	writer <- &anyseq.Batch{
		Packed:  c.MakeVectorData([]float64{-1, -2, 7, 8}),
		Present: []bool{false, false, true, true},
	}
	writer <- &anyseq.Batch{
		Packed:  c.MakeVectorData([]float64{-3, -4}),
		Present: []bool{false, false, true, false},
	}
	close(writer)

	out := lazyseq.ReduceTape(tape, []bool{true, true, false, true}).ReadTape(0, -1)
	mustRead(t, &anyseq.Batch{
		Packed:  c.MakeVectorData([]float64{1, 2, 5, 6}),
		Present: []bool{true, false, false, true},
	}, out)
	mustRead(t, &anyseq.Batch{
		Packed:  c.MakeVectorData([]float64{7, 8}),
		Present: []bool{false, false, false, true},
	}, out)
	mustRead(t, nil, out)
}
