package lazyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestMapN(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	const inSize = 3

	seqs := []anyseq.Seq{
		testSeqsLen(c, inSize, 1, 2, 0, 3, 3),
		testSeqsLen(c, inSize, 1, 2, 0, 3, 3),
	}

	f := func(n int, reses ...anydiff.Res) anydiff.Res {
		return anydiff.Scale(
			anydiff.Sub(reses[0], reses[1]),
			reses[0].Output().Creator().MakeNumeric(float64(n)),
		)
	}

	testEquivalent(t, func() anyseq.Seq {
		var lazySeqs []Rereader
		for _, s := range seqs {
			lazySeqs = append(lazySeqs, Lazify(s))
		}
		return Unlazify(MapN(f, lazySeqs...))
	}, func() anyseq.Seq {
		return anyseq.MapN(f, seqs...)
	})
}

func TestMapNReread(t *testing.T) {
	// Only way to test Reread is with something that
	// requires it.

	c := anyvec64.DefaultCreator{}
	const inSize = 3
	const outSize = 2

	seqs := []anyseq.Seq{
		testSeqsLen(c, inSize, 1, 7, 0, 3, 3),
		testSeqsLen(c, inSize, 1, 7, 0, 3, 3),
	}

	block := anyrnn.NewLSTM(c, inSize, outSize)

	f := func(n int, reses ...anydiff.Res) anydiff.Res {
		return anydiff.Scale(
			anydiff.Sub(reses[0], reses[1]),
			reses[0].Output().Creator().MakeNumeric(float64(n)),
		)
	}

	testEquivalent(t, func() anyseq.Seq {
		var lazySeqs []Rereader
		for _, s := range seqs {
			lazySeqs = append(lazySeqs, Lazify(s))
		}
		seq := MapN(f, lazySeqs...)
		return Unlazify(FixedHSM(3, seq, block))
	}, func() anyseq.Seq {
		seq := anyseq.MapN(f, seqs...)
		return anyrnn.Map(seq, block)
	})
}
