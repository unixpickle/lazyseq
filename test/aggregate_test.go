package test

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestMean(t *testing.T) {
	const inSize = 3
	c := anyvec64.DefaultCreator{}
	inSeqs := testSeqs(c, inSize)

	count := 0
	for _, seq := range anyseq.SeparateSeqs(inSeqs.Output()) {
		count += len(seq)
	}

	actualFunc := func() anydiff.Res {
		return lazyseq.Mean(lazyseq.Lazify(inSeqs))
	}
	expectedFunc := func() anydiff.Res {
		return anydiff.Scale(anyseq.Sum(inSeqs),
			inSeqs.Creator().MakeNumeric(1/float64(count)))
	}
	testEquivalentRes(t, actualFunc, expectedFunc)
}

func TestSumEach(t *testing.T) {
	const inSize = 3
	c := anyvec64.DefaultCreator{}
	inSeqs := testSeqs(c, inSize)

	actualFunc := func() anydiff.Res {
		return lazyseq.SumEach(lazyseq.Lazify(inSeqs))
	}
	expectedFunc := func() anydiff.Res {
		return anyseq.SumEach(inSeqs)
	}
	testEquivalentRes(t, actualFunc, expectedFunc)
}

func TestSum(t *testing.T) {
	const inSize = 3
	c := anyvec64.DefaultCreator{}
	inSeqs := testSeqs(c, inSize)

	actualFunc := func() anydiff.Res {
		return lazyseq.Sum(lazyseq.Lazify(inSeqs))
	}
	expectedFunc := func() anydiff.Res {
		return anyseq.Sum(inSeqs)
	}
	testEquivalentRes(t, actualFunc, expectedFunc)
}
