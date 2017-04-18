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
