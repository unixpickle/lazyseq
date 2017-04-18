package test

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestTailEquiv(t *testing.T) {
	const inSize = 3
	c := anyvec64.DefaultCreator{}
	inSeqs := testSeqs(c, inSize)
	actualFunc := func() anydiff.Res {
		return lazyseq.Tail(lazyseq.Lazify(inSeqs))
	}
	expectedFunc := func() anydiff.Res {
		return anyseq.Tail(inSeqs)
	}
	testEquivalentRes(t, actualFunc, expectedFunc)
}
