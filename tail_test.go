package lazyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestTailEquiv(t *testing.T) {
	const inSize = 3
	c := anyvec64.DefaultCreator{}
	inSeqs := testSeqs(c, inSize)
	actualFunc := func() anyseq.Seq {
		return anyseq.ResSeq(c, []*anyseq.ResBatch{
			&anyseq.ResBatch{
				Packed:  Tail(Lazify(inSeqs)),
				Present: []bool{true},
			},
		})
	}
	expectedFunc := func() anyseq.Seq {
		return anyseq.ResSeq(c, []*anyseq.ResBatch{
			&anyseq.ResBatch{
				Packed:  anyseq.Tail(inSeqs),
				Present: []bool{true},
			},
		})
	}
	testEquivalent(t, actualFunc, expectedFunc)
}
