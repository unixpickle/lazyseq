package lazyrnn

import (
	"fmt"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestFixedHSMEquiv(t *testing.T) {
	const inSize = 3
	const outSize = 2

	c := anyvec64.DefaultCreator{}

	block := anyrnn.NewLSTM(c, inSize, outSize)

	for interval := 1; interval < 10; interval++ {
		t.Run(fmt.Sprintf("Interval%d", interval), func(t *testing.T) {
			inSeqs := testSeqs(c, inSize)
			actualFunc := func() anyseq.Seq {
				return Unlazify(FixedHSM(interval, Lazify(inSeqs), block))
			}
			expectedFunc := func() anyseq.Seq {
				return anyrnn.Map(inSeqs, block)
			}
			testEquivalent(t, actualFunc, expectedFunc)
		})
	}
}
