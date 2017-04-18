package lazyrnn

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/essentials"
)

func TestPack(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	const inSize = 2

	seqs := []anyseq.Seq{
		testSeqsLen(c, inSize, 1, 0, 5),
		testSeqsLen(c, inSize, 7, 1),
		testSeqsLen(c, inSize, 3, 4, 1, 3),
		testSeqsLen(c, inSize, 5),
	}

	testEquivalent(t, func() anyseq.Seq {
		var lazySeqs []Seq
		for _, s := range seqs {
			lazySeqs = append(lazySeqs, Lazify(s))
		}
		return Unlazify(Pack(c, lazySeqs))
	}, func() anyseq.Seq {
		return packAnyseq(c, seqs)
	})
}

func TestPackRereader(t *testing.T) {
	// To test this, we have to use something that will
	// actually call Reread().

	const inSize = 3
	const outSize = 2

	c := anyvec64.DefaultCreator{}
	seqs := []anyseq.Seq{
		testSeqsLen(c, inSize, 1, 0, 5),
		testSeqsLen(c, inSize, 7, 1),
		testSeqsLen(c, inSize, 3, 4, 1, 3),
		testSeqsLen(c, inSize, 5),
	}
	block := anyrnn.NewLSTM(c, inSize, outSize)

	testEquivalent(t, func() anyseq.Seq {
		var lazySeqs []Rereader
		for _, s := range seqs {
			lazySeqs = append(lazySeqs, Lazify(s))
		}
		return Unlazify(FixedHSM(3, true, PackRereader(c, lazySeqs), block))
	}, func() anyseq.Seq {
		return anyrnn.Map(packAnyseq(c, seqs), block)
	})
}

func TestPackTape(t *testing.T) {
	c := anyvec64.DefaultCreator{}

	tape1, writer1 := ReferenceTape()
	tape2, writer2 := ReferenceTape()
	tape3, writer3 := ReferenceTape()

	joined := PackTape([]Tape{tape1, tape2, tape3})

	inBatches := []*anyseq.Batch{
		{Present: []bool{true, false, true}},
		{Present: []bool{true, false, false}},
		{Present: []bool{true, false, false}},
		{Present: []bool{false, false, false}},

		{Present: []bool{true, true}},
		{Present: []bool{false, true}},
	}
	for _, b := range inBatches {
		b.Packed = c.MakeVector(b.NumPresent() * 5)
		anyvec.Rand(b.Packed, anyvec.Normal, nil)
	}

	outBatches := []*anyseq.Batch{}
	for i := 0; i < 2; i++ {
		outBatches = append(outBatches, &anyseq.Batch{
			Present: append(append([]bool{}, inBatches[i].Present...),
				inBatches[i+4].Present...),
			Packed: c.Concat(inBatches[i].Packed, inBatches[i+4].Packed),
		})
	}
	for i := 2; i < 4; i++ {
		outBatches = append(outBatches, &anyseq.Batch{
			Present: append(append([]bool{}, inBatches[i].Present...),
				false, false),
			Packed: inBatches[i].Packed,
		})
	}

	out1 := joined.ReadTape(0, 1)
	out2 := joined.ReadTape(1, -1)

	writer3 <- inBatches[4]
	writer3 <- inBatches[5]
	writer1 <- inBatches[0]
	close(writer2)

	mustRead(t, outBatches[0], out1)
	mustNotRead(t, out2)

	writer1 <- inBatches[1]
	mustRead(t, nil, out1)
	mustRead(t, outBatches[1], out2)

	writer1 <- inBatches[2]
	close(writer3)

	mustRead(t, outBatches[2], out2)

	writer1 <- inBatches[3]
	mustRead(t, outBatches[3], out2)

	close(writer1)

	mustRead(t, nil, out2)
}

func packAnyseq(c anyvec.Creator, seqs []anyseq.Seq) anyseq.Seq {
	if len(seqs) == 0 {
		return anyseq.ConstSeqList(c, nil)
	} else if len(seqs) == 1 {
		return seqs[0]
	} else if len(seqs) == 2 {
		return newAnyseqPackedPair(seqs[0], seqs[1])
	} else {
		firstHalf := packAnyseq(c, seqs[:len(seqs)/2])
		secondHalf := packAnyseq(c, seqs[len(seqs)/2:])
		return packAnyseq(c, []anyseq.Seq{firstHalf, secondHalf})
	}
}

type anyseqPackedPair struct {
	Seqs [2]anyseq.Seq
	Outs []*anyseq.Batch
}

func newAnyseqPackedPair(s1, s2 anyseq.Seq) *anyseqPackedPair {
	res := &anyseqPackedPair{Seqs: [2]anyseq.Seq{s1, s2}}
	c := s1.Creator()

	lanes1 := res.numLanes(0)
	lanes2 := res.numLanes(1)

	out1 := s1.Output()
	out2 := s2.Output()
	outLen := essentials.MaxInt(len(out1), len(out2))
	for i := 0; i < outLen; i++ {
		batch := &anyseq.Batch{
			Present: make([]bool, lanes1+lanes2),
			Packed:  c.MakeVector(0),
		}
		if i < len(out1) {
			batch1 := out1[i]
			batch.Packed = c.Concat(batch.Packed, batch1.Packed)
			copy(batch.Present, batch1.Present)
		}
		if i < len(out2) {
			batch2 := out2[i]
			batch.Packed = c.Concat(batch.Packed, batch2.Packed)
			copy(batch.Present[lanes1:], batch2.Present)
		}
		res.Outs = append(res.Outs, batch)
	}

	return res
}

func (a *anyseqPackedPair) Creator() anyvec.Creator {
	return a.Seqs[0].Creator()
}

func (a *anyseqPackedPair) Output() []*anyseq.Batch {
	return a.Outs
}

func (a *anyseqPackedPair) Vars() anydiff.VarSet {
	return anydiff.MergeVarSets(a.Seqs[0].Vars(), a.Seqs[1].Vars())
}

func (a *anyseqPackedPair) Propagate(u []*anyseq.Batch, grad anydiff.Grad) {
	var firstU, secondU []*anyseq.Batch
	lanes1 := a.numLanes(0)
	for _, batch := range u {
		vecSize := batch.Packed.Len() / batch.NumPresent()
		batch1 := &anyseq.Batch{Present: batch.Present[:lanes1]}
		size1 := batch1.NumPresent() * vecSize
		batch1.Packed = batch.Packed.Slice(0, size1)
		batch2 := &anyseq.Batch{
			Packed:  batch.Packed.Slice(size1, batch.Packed.Len()),
			Present: batch.Present[lanes1:],
		}
		if batch1.NumPresent() > 0 {
			firstU = append(firstU, batch1)
		}
		if batch2.NumPresent() > 0 {
			secondU = append(secondU, batch2)
		}
	}
	a.Seqs[0].Propagate(firstU, grad)
	a.Seqs[1].Propagate(secondU, grad)
}

func (a *anyseqPackedPair) numLanes(seqIdx int) int {
	seq := a.Seqs[seqIdx]
	if len(seq.Output()) == 0 {
		return 0
	}
	return len(seq.Output()[0].Present)
}
