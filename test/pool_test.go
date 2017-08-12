package test

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestPoolToVec(t *testing.T) {
	const inSize = 3
	c := anyvec64.DefaultCreator{}
	inSeqs := testSeqs(c, inSize)

	count := 0
	for _, seq := range anyseq.SeparateSeqs(inSeqs.Output()) {
		count += len(seq)
	}

	actualFunc := func() anydiff.Res {
		return lazyseq.PoolToVec(lazyseq.Lazify(inSeqs),
			func(seq lazyseq.Rereader) anydiff.Res {
				return lazyseq.Mean(seq)
			})
	}
	expectedFunc := func() anydiff.Res {
		return lazyseq.Mean(lazyseq.Lazify(inSeqs))
	}
	testEquivalentRes(t, actualFunc, expectedFunc)
}
