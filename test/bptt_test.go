package test

import (
	"fmt"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
)

func TestBPTTEquiv(t *testing.T) {
	const inSize = 3
	const outSize = 2

	c := anyvec64.DefaultCreator{}

	block := anyrnn.NewLSTM(c, inSize, outSize)

	for interval := 1; interval < 10; interval++ {
		for _, lazy := range []bool{false, true} {
			t.Run(fmt.Sprintf("Interval%d:%v", interval, lazy), func(t *testing.T) {
				inSeqs := testSeqs(c, inSize)
				actualFunc := func() anyseq.Seq {
					return lazyseq.Unlazify(lazyrnn.BPTT(lazyseq.Lazify(inSeqs),
						block))
				}
				expectedFunc := func() anyseq.Seq {
					return anyrnn.Map(inSeqs, block)
				}
				testEquivalent(t, actualFunc, expectedFunc)
			})
		}
	}
}
