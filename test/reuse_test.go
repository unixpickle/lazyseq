package test

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/lazyseq"
)

func TestReuse(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	seqs := testSeqs(c, 3)

	reuser := lazyseq.MakeReuser(lazyseq.Lazify(seqs))

	f := func() anyseq.Seq {
		mapFunc := func(in anydiff.Res, n int) anydiff.Res {
			return anydiff.Scale(in, in.Output().Creator().MakeNumeric(0.7))
		}
		res := lazyseq.Unlazify(lazyseq.Map(reuser, mapFunc))
		reuser.Reuse()
		return res
	}

	testEquivalent(t, f, f)
}
