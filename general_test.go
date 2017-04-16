package lazyrnn

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

// testSeqs generates test sequences of varying lengths.
//
// The resulting sequence will depend on one variable per
// timestep, i.e. it will not be constant.
func testSeqs(c anyvec.Creator, inSize int) anyseq.Seq {
	const numSeqs = 8

	lengths := rand.Perm(numSeqs)

	// Ensure that two sequences are the same length,
	// thus catching potential edge-cases.
	lengths[0] = lengths[2]
	lengths[3] = lengths[5]

	return testSeqsLen(c, inSize, lengths...)
}

func testSeqsLen(c anyvec.Creator, inSize int, lengths ...int) anyseq.Seq {
	var seqs [][]anyvec.Vector
	for i := 0; i < len(lengths); i++ {
		var seq []anyvec.Vector
		for j := 0; j < lengths[i]; j++ {
			vec := c.MakeVector(inSize)
			anyvec.Rand(vec, anyvec.Normal, nil)
			seq = append(seq, vec)
		}
		seqs = append(seqs, seq)
	}

	joined := anyseq.ConstSeqList(c, seqs)

	resBatches := make([]*anyseq.ResBatch, len(joined.Output()))
	for i, x := range joined.Output() {
		resBatches[i] = &anyseq.ResBatch{
			Packed:  anydiff.NewVar(x.Packed),
			Present: x.Present,
		}
	}

	return anyseq.ResSeq(c, resBatches)
}

// testEquivalent ensures that two ways of producing an
// anyseq.Seq are equivalent.
func testEquivalent(t *testing.T, actual, expected func() anyseq.Seq) {
	t.Run("Vars", func(t *testing.T) {
		testVarEquivalence(t, actual, expected)
	})
	t.Run("Out", func(t *testing.T) {
		testOutEquivalence(t, actual, expected)
	})
	t.Run("Grad", func(t *testing.T) {
		testGradEquivalence(t, actual, expected)
	})
}

// testEquivalentRes is like testEquivalent, but for
// functions which produce vectors.
func testEquivalentRes(t *testing.T, actual, expected func() anydiff.Res) {
	c := expected().Output().Creator()
	actualFunc := func() anyseq.Seq {
		return anyseq.ResSeq(c, []*anyseq.ResBatch{
			&anyseq.ResBatch{
				Packed:  actual(),
				Present: []bool{true},
			},
		})
	}
	expectedFunc := func() anyseq.Seq {
		return anyseq.ResSeq(c, []*anyseq.ResBatch{
			&anyseq.ResBatch{
				Packed:  expected(),
				Present: []bool{true},
			},
		})
	}
	testEquivalent(t, actualFunc, expectedFunc)
}

func testVarEquivalence(t *testing.T, actual, expected func() anyseq.Seq) {
	vars1 := actual().Vars()
	vars2 := expected().Vars()
	if len(vars1) != len(vars2) {
		t.Error("variable mismatch")
	} else {
		for x := range vars1 {
			if !vars2.Has(x) {
				t.Error("variable mismatch")
			}
		}
	}
}

func testOutEquivalence(t *testing.T, actual, expected func() anyseq.Seq) {
	actOut := actual().Output()
	expOut := expected().Output()
	if len(actOut) != len(expOut) {
		t.Errorf("output length: expected %d got %d", len(expOut), len(actOut))
		return
	}
	for i, actBatch := range actOut {
		expBatch := expOut[i]
		if !reflect.DeepEqual(actBatch.Present, expBatch.Present) {
			t.Errorf("present mismatch: time %d: expected %v got %v", i,
				expBatch.Present, actBatch.Present)
			return
		}
		v1 := actBatch.Packed.Copy()
		v1.Sub(expBatch.Packed)
		maxDiff := anyvec.AbsMax(v1).(float64)
		if maxDiff > 1e-4 {
			t.Errorf("output mismatch: time %d: expected %v got %v", i,
				expBatch.Packed.Data(), actBatch.Packed.Data())
			return
		}
	}
}

func testGradEquivalence(t *testing.T, actual, expected func() anyseq.Seq) {
	t.Run("AllVars", func(t *testing.T) {
		actGrad := computeGradient(actual(), nil)
		expGrad := computeGradient(expected(), nil)
		gradientsEquivalent(t, actGrad, expGrad)
	})
	t.Run("SingleVar", func(t *testing.T) {
		for v := range actual().Vars() {
			vs := anydiff.NewVarSet(v)
			actGrad := computeGradient(actual(), vs)
			expGrad := computeGradient(expected(), vs)
			gradientsEquivalent(t, actGrad, expGrad)
		}
	})
}

func computeGradient(seq anyseq.Seq, vars anydiff.VarSet) anydiff.Grad {
	if vars == nil {
		vars = seq.Vars()
	}

	grad := anydiff.NewGrad(vars.Slice()...)

	upstreamGen := rand.New(rand.NewSource(1337))
	upstream := make([]*anyseq.Batch, len(seq.Output()))
	for i, x := range seq.Output() {
		data := make([]float64, x.Packed.Len())
		for i := range data {
			data[i] = upstreamGen.NormFloat64()
		}
		upstream[i] = &anyseq.Batch{
			Present: x.Present,
			Packed:  x.Packed.Creator().MakeVectorData(data),
		}
	}

	seq.Propagate(upstream, grad)
	return grad
}

func gradientsEquivalent(t *testing.T, actGrad, expGrad anydiff.Grad) {
	for variable, vec := range actGrad {
		expVec := expGrad[variable]
		if expVec == nil {
			t.Error("excess variable")
			continue
		}
		diff := expVec.Copy()
		diff.Sub(vec)
		maxDiff := anyvec.AbsMax(diff).(float64)
		if maxDiff > 1e-4 {
			t.Errorf("gradient mismatch: expected %v got %v", expVec.Data(),
				vec.Data())
			return
		}
	}
}
